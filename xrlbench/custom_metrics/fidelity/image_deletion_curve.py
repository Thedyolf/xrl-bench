
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _is_channel_first(x: np.ndarray) -> bool:
    # Heuristic: (N, C, H, W) where C in {1,3,4} and last dim not in {1,3,4}
    if x.ndim != 4:
        return False
    c = x.shape[1]
    last = x.shape[-1]
    return (c in (1, 3, 4)) and (last not in (1, 3, 4))


def _as_channel_last(x: np.ndarray) -> np.ndarray:
    # (N,C,H,W) -> (N,H,W,C)
    if _is_channel_first(x):
        return np.transpose(x, (0, 2, 3, 1))
    return x


def _as_channel_first(x: np.ndarray) -> np.ndarray:
    # (N,H,W,C) -> (N,C,H,W)
    if _is_channel_first(x):
        return x
    return np.transpose(x, (0, 3, 1, 2))


def _get_model_from_environment(environment: Any):
    """
    Try to extract a torch model from `environment`.
    Supports several common patterns:
      - environment.model
      - environment.agent.model
      - environment.policy
      - callable environment (rare)
    """
    for attr in ("model", "policy"):
        if hasattr(environment, attr):
            return getattr(environment, attr)

    # Some wrappers store agent/model in nested objects
    if hasattr(environment, "agent") and hasattr(environment.agent, "model"):
        return environment.agent.model

    if callable(environment):
        return environment

    raise AttributeError(
        "ImageDeletionCurve could not locate a model in `environment`. "
        "Expected attributes like `.model`, `.policy`, or `.agent.model`."
    )


def _infer_device(model: Any) -> torch.device:
    try:
        params = list(model.parameters())
        if params:
            return params[0].device
    except Exception:
        pass
    return torch.device("cpu")


def _predict_action_scores(
    model: Any,
    x_img: np.ndarray,
    y_action: np.ndarray,
    score_mode: str = "logit",
) -> np.ndarray:
    """
    Returns a per-sample scalar score for the provided target action y_action.

    score_mode:
      - "logit": take raw model output for the action index
      - "prob":  apply softmax and take probability for the action index
    """
    if x_img.ndim != 4:
        raise ValueError(f"Expected X with ndim==4, got shape={x_img.shape}")

    # Model convention is most often channel-first for torch (N,C,H,W).
    x_cf = _as_channel_first(x_img).astype(np.float32, copy=False)

    # Ensure contiguous layout so that model code using .view(...) does not fail.
    x_cf = np.ascontiguousarray(x_cf)

    device = _infer_device(model)
    xt = torch.from_numpy(x_cf).to(device)
    xt = xt.contiguous()

    with torch.no_grad():
        out = model(xt)

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Model output must be torch.Tensor, got {type(out)!r}")

    if out.ndim != 2:
        # Some models output extra dims; try flattening to (N,A)
        out = out.view(out.shape[0], -1)

    y = torch.as_tensor(y_action, device=out.device, dtype=torch.long).view(-1, 1)

    if score_mode == "prob":
        probs = torch.softmax(out, dim=1)
        scores = probs.gather(1, y).squeeze(1)
    elif score_mode == "logit":
        scores = out.gather(1, y).squeeze(1)
    else:
        raise ValueError("score_mode must be one of {'logit','prob'}")

    return scores.detach().cpu().numpy()



def _baseline_image(x_img: np.ndarray, baseline: str) -> np.ndarray:
    """
    Per-sample baseline image.
    baseline:
      - "zero": all zeros
      - "mean": per-channel mean over pixels
      - "median": per-channel median over pixels
    """
    x_cl = _as_channel_last(x_img)  # (N,H,W,C)
    N, H, W, C = x_cl.shape

    if baseline == "zero":
        return np.zeros((N, H, W, C), dtype=x_cl.dtype)

    flat = x_cl.reshape(N, H * W, C)  # (N,P,C)
    if baseline == "mean":
        b = flat.mean(axis=1, keepdims=True)  # (N,1,C)
    elif baseline == "median":
        b = np.median(flat, axis=1, keepdims=True)  # (N,1,C)
    else:
        raise ValueError("baseline must be one of {'zero','mean','median'}")

    return np.broadcast_to(b.reshape(N, 1, 1, C), (N, H, W, C)).copy()


@dataclass
class ImageDeletionCurve:
    """
    Incremental deletion curve for image environments (DAUC-style).

    Procedure:
      1) Rank pixels by saliency (descending).
      2) Progressively replace top-ranked pixels with a baseline (zero/mean/median).
      3) Record model score at each deletion fraction.
      4) Return curve + AUC (trapezoidal rule).

    Notes:
      - This is "incremental": it updates a running perturbed image by applying only the
        additional pixels needed for the next fraction.
      - Saliency values are used only for ranking (standard DAUC/IAUC behavior).
    """
    environment: Any
    baseline: str = "mean"
    fractions: Optional[np.ndarray] = None
    score_mode: str = "logit"  # "logit" (RL/Q-values) or "prob" (classification)

    def __post_init__(self):
        if self.fractions is None:
            self.fractions = np.linspace(0.0, 1.0, 9, dtype=float)

        self.fractions = np.asarray(self.fractions, dtype=float)
        if self.fractions.ndim != 1 or self.fractions.size < 2:
            raise ValueError("fractions must be a 1D array with at least 2 values.")
        if np.any(self.fractions < 0.0) or np.any(self.fractions > 1.0):
            raise ValueError("fractions must be in [0,1].")
        if np.any(np.diff(self.fractions) < 0):
            raise ValueError("fractions must be non-decreasing.")

        self.model = _get_model_from_environment(self.environment)

    def evaluate(self, X: Any, y: Any, importance: Any) -> Dict[str, Any]:
        """
        X: (N,C,H,W) or (N,H,W,C)
        y: (N,) action indices
        importance: (N,H,W) per-pixel saliency (as produced by prepare_importance_for_image_metrics)

        Returns:
            {
                "fractions": 1D array of deletion fractions in [0,1],
                "curve": mean normalized deletion curve in [0,1],
                "auc": scalar AUC over normalized curve (lower is better)
            }

        Normalization:
            For each sample n:
              - s_n(0)  = score at fraction 0  (original image)
              - s_n(end) = score at last fraction (typically fully baseline image)

            We define:
              h_n(f) = (s_n(f) - s_n(end)) / (s_n(0) - s_n(end) + eps)

            So:
              h_n(0)   ≈ 1
              h_n(end) ≈ 0

            Then we average h_n(f) over n to get the final curve.
        """
        X_np = _to_numpy(X)
        y_np = _to_numpy(y).astype(int)
        imp = _to_numpy(importance)

        if X_np.ndim != 4:
            raise ValueError(f"Expected X.ndim==4, got {X_np.ndim} with shape {X_np.shape}")
        if imp.ndim != 3:
            raise ValueError(f"Expected importance shape (N,H,W), got {imp.shape}")

        X_cl = _as_channel_last(X_np).copy()  # (N,H,W,C)
        N, H, W, C = X_cl.shape
        P = H * W

        if imp.shape[0] != N or imp.shape[1] != H or imp.shape[2] != W:
            raise ValueError(
                f"importance must match X spatial dims. "
                f"Got X (N,H,W)=({N},{H},{W}) but importance={imp.shape}"
            )

        # Per-sample ranking of pixels by descending saliency
        # --- New: choose which pixels to delete based on POSITIVE contributions only ---

        # imp: (N,H,W) signed importance map
        flat_imp = imp.reshape(N, P)

        # Positive part: evidence in favour of the selected action
        pos_imp = np.maximum(flat_imp, 0.0)

        # If a sample has no positive contributions at all (all <= 0),
        # fall back to absolute values to still get an ordering.
        max_pos = pos_imp.max(axis=1)  # (N,)
        no_pos_mask = max_pos <= 0.0

        flat_for_order = pos_imp.copy()
        if np.any(no_pos_mask):
            flat_for_order[no_pos_mask] = np.abs(flat_imp[no_pos_mask])

        # Rank pixels by descending "evidence for the action"
        order = np.argsort(-flat_for_order, axis=1)  # (N,P)

        # Baseline image in channel-last format
        base = _baseline_image(X_cl, self.baseline)  # (N,H,W,C)

        # Number of pixels to delete per fraction
        counts = np.clip(np.round(self.fractions * P).astype(int), 0, P)
        counts = np.maximum.accumulate(counts)  # enforce non-decreasing even after rounding

        x_cur = X_cl.copy()
        prev = 0

        n_idx = np.arange(N)
        all_scores = []  # will hold list of arrays of shape (N,)

        for cnt in counts:
            if cnt > prev:
                # Newly deleted pixels indices: [prev:cnt]
                delta = order[:, prev:cnt]  # (N, cnt-prev)
                hh = (delta // W).astype(int)
                ww = (delta % W).astype(int)

                # Apply baseline to these pixels
                x_cur[n_idx[:, None], hh, ww, :] = base[n_idx[:, None], hh, ww, :]

                prev = cnt

            # Model scores for target actions for this fraction
            scores = _predict_action_scores(self.model, x_cur, y_np, score_mode=self.score_mode)
            all_scores.append(scores)  # shape (N,)

        # Stack to (N, T) where T = len(fractions)
        S = np.stack(all_scores, axis=1).astype(float)  # (N, T)

        # Per-sample normalization to [0,1]
        s0 = S[:, 0]      # scores at fraction 0
        sL = S[:, -1]     # scores at last fraction
        denom = s0 - sL
        eps = 1e-8

        H = np.zeros_like(S, dtype=float)

        good = np.abs(denom) > eps
        if np.any(good):
            num = S[good] - sL[good, None]
            H[good] = num / (denom[good, None] + eps)

        # Degenerate samples where s0 ≈ sL: treat as uninformative flat curve 0.5
        bad = ~good
        if np.any(bad):
            H[bad] = 0.5

        # Clip to [0,1] to avoid numerical issues
        H = np.clip(H, 0.0, 1.0)

        # Mean normalized curve over samples
        curve = H.mean(axis=0)

        # AUC over fractions (now in [0,1])
        auc = float(np.trapz(curve, self.fractions))

        return {
            "fractions": self.fractions,
            "curve": curve,
            "auc": auc,
        }
