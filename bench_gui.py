# bench_gui.py
# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import h5py

import tkinter as tk
from tkinter import ttk, messagebox

import torch
import warnings

from xrlbench.explainers import Explainer
from xrlbench.evaluator import Evaluator
from xrlbench.environments import Environment

# ----------------------------
# Image explainers
# ----------------------------
from xrlbench.custom_explainers.sarfa import ImageSARFA
from xrlbench.custom_explainers.perturbation_saliency import ImagePerturbationSaliency
from xrlbench.custom_explainers.deep_shap import ImageDeepSHAP
from xrlbench.custom_explainers.gradient_shap import ImageGradientSHAP
from xrlbench.custom_explainers.integrated_gradient import ImageIntegratedGradient

# ----------------------------
# Metrics: tabular + image variants
# ----------------------------
from xrlbench.custom_metrics.fidelity.aim import AIM, ImageAIM
from xrlbench.custom_metrics.fidelity.aum import AUM, ImageAUM
from xrlbench.custom_metrics.fidelity.pgi import PGI, ImagePGI
from xrlbench.custom_metrics.fidelity.pgu import PGU, ImagePGU
from xrlbench.custom_metrics.stability.ris import RIS, ImageRIS
from xrlbench.custom_metrics.fidelity.deletion_curve import DeletionCurve
from xrlbench.custom_metrics.fidelity.image_deletion_curve import ImageDeletionCurve

warnings.filterwarnings("ignore", message="Gym has been unmaintained since 2022.*")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message="WARN: The environment .* is out of date.*")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

# --------------------------------------------------------------------
# Environment types
# --------------------------------------------------------------------
TABULAR_ENVS = {"lunarLander", "cartPole"}
IMAGE_ENVS = {"pong", "breakOut"}


def is_image_env(env_name: str) -> bool:
    return env_name in IMAGE_ENVS


def is_tabular_env(env_name: str) -> bool:
    return env_name in TABULAR_ENVS


# --------------------------------------------------------------------
# Seeding / helpers
# --------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed_all(seed)


def _safe_float(x):
    if isinstance(x, (np.floating, float, int)):
        return float(x)
    return x

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def parse_deletion_fractions(spec: str) -> np.ndarray:
    """
    Accepts '0, 1, 9' meaning linspace(0,1,9)
    Also accepts explicit list like '0, 0.1, 0.2, 0.5, 1'
    """
    parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
    if len(parts) < 2:
        raise ValueError("Deletion fractions must be 'start,end,num' or an explicit list.")

    nums = [float(p) for p in parts]

    if len(nums) == 3:
        start, end, n = nums
        n_int = int(round(n))
        if n_int <= 1:
            raise ValueError("For 'start,end,num', num must be >= 2.")
        return np.linspace(start, end, n_int)

    return np.array(nums, dtype=float)


def get_available_environments() -> list[str]:
    return ["lunarLander", "cartPole", "pong", "breakOut"]


# --------------------------------------------------------------------
# Image dataset loaders
# --------------------------------------------------------------------
def _load_image_h5(path: str, log) -> tuple[np.ndarray, np.ndarray]:
    """
    Expects datasets 'observations' and 'actions' in the H5 file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"H5 file not found: {path}")

    with h5py.File(path, "r") as f:
        if "observations" not in f or "actions" not in f:
            raise KeyError(
                f"H5 file {path} must contain 'observations' and 'actions'. "
                f"Found keys: {list(f.keys())}"
            )
        X_all = f["observations"][:]
        y_all = f["actions"][:]

    log(f"[Worker] Loaded H5: {path}")
    log(f"  X_all shape: {X_all.shape}")
    log(f"  y_all shape: {y_all.shape}")
    return X_all, y_all

class ImageExplainerReducer:
    """
    Wraps an image explainer so that .explain() always returns a per-sample
    saliency map of shape (N, H, W), regardless of the explainer's native output.
    """
    def __init__(self, base_explainer):
        self.base = base_explainer

    def explain(self, *args, **kwargs):
        raw = self.base.explain(*args, **kwargs)

        # Try to infer the current X from the explainer (RIS often perturbs X internally)
        X_ref = None
        for attr in ("X", "x", "states", "state"):
            if hasattr(self.base, attr):
                X_ref = getattr(self.base, attr)
                break

        if X_ref is None:
            # Fallback: if base explainer doesn't expose X, we cannot reduce robustly
            raise AttributeError(
                "Cannot infer X from explainer; cannot reduce importance for ImageRIS."
            )

        X_ref = _to_numpy(X_ref)
        return prepare_importance_for_image_metrics(raw, X_ref)

    def __getattr__(self, name):
        # Delegate all other attributes/methods to base explainer
        return getattr(self.base, name)


def load_image_dataset_for_env(env_name: str, log) -> tuple[np.ndarray, np.ndarray]:
    if env_name == "pong":
        path = os.path.join(".", "data", "Pong_dataset.h5")
    elif env_name == "breakOut":
        path = os.path.join(".", "data", "BreakOut_dataset.h5")
    else:
        raise ValueError(f"Unknown image environment for H5 loading: {env_name}")
    return _load_image_h5(path, log)


# --------------------------------------------------------------------
# Method / metric support rules
# --------------------------------------------------------------------
METHODS_ALL = [
    "tabularShap",
    "tabularLime",
    "sarfa",
    "perturbationSaliency",
    "deepShap",
    "gradientShap",
    "integratedGradient",
]

METRICS_ALL = [
    "AIM",
    "AUM",
    "PGI",
    "PGU",
    "RIS",
    "DeletionCurve",
    "ImageDeletionCurve",
]


def method_supported(env_name: str, method: str) -> bool:
    if is_tabular_env(env_name):
        # Grey out gradientShap for tabular envs
        if method == "gradientShap":
            return False
        return method in {
            "tabularShap",
            "tabularLime",
            "sarfa",
            "perturbationSaliency",
            "deepShap",
            "integratedGradient",
        }

    if is_image_env(env_name):
        # Only image-capable explainers
        return method in {
            "sarfa",
            "perturbationSaliency",
            "deepShap",
            "gradientShap",
            "integratedGradient",
        }

    return False


def metric_supported(env_name: str, metric: str) -> bool:
    if is_tabular_env(env_name):
        # Full set for tabular, including PGI
        return metric in {"AIM", "AUM", "PGI", "PGU", "RIS", "DeletionCurve"}

    if is_image_env(env_name):
        # Disable PGI for image envs: keep it only for tabular
        # Still allow AIM/AUM/PGU/RIS via Image* metrics + ImageDeletionCurve
        return metric in {"AIM", "AUM", "PGU", "RIS", "ImageDeletionCurve"}

    return False



# --------------------------------------------------------------------
# Image explainer builder
# --------------------------------------------------------------------
def build_image_explainer(method: str, X: np.ndarray, y: np.ndarray, model):
    method = method.strip()
    if method == "sarfa":
        return ImageSARFA(X=X, y=y, model=model)
    if method == "perturbationSaliency":
        return ImagePerturbationSaliency(X=X, y=y, model=model)
    if method == "deepShap":
        return ImageDeepSHAP(X=X, y=y, model=model)
    if method == "gradientShap":
        return ImageGradientSHAP(X=X, y=y, model=model)
    if method == "integratedGradient":
        return ImageIntegratedGradient(X=X, y=y, model=model)
    raise ValueError(f"No image explainer implemented for method='{method}'.")


# --------------------------------------------------------------------
# Importance prep for image metrics
# --------------------------------------------------------------------
def prepare_importance_for_image_metrics(importance_raw, X: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary image attributions into a (N, H, W) saliency map.

    This version PRESERVES SIGN:
      - + values: evidence in favour of the current action
      - - values: evidence against the current action
    """
    imp = _to_numpy(importance_raw)
    X_np = _to_numpy(X)
    N = X_np.shape[0]

    if X_np.ndim != 4:
        raise ValueError(f"Expected X.ndim==4 for image envs; got {X_np.ndim}.")

    if X_np.shape[1] in (1, 3, 4) and X_np.shape[-1] not in (1, 3, 4):
        # (N,C,H,W)
        H, W = X_np.shape[2], X_np.shape[3]
    else:
        # (N,H,W,C)
        H, W = X_np.shape[1], X_np.shape[2]

    P = H * W

    # Move batch axis to front
    if imp.shape[0] != N:
        axesN = [ax for ax, s in enumerate(imp.shape) if s == N]
        if not axesN:
            raise ValueError(f"Could not find batch axis N={N} in importance shape {imp.shape}.")
        if axesN[0] != 0:
            imp = np.moveaxis(imp, axesN[0], 0)

    F_total = int(np.prod(imp.shape[1:]))
    if F_total % P != 0:
        raise ValueError(
            f"Importance total features={F_total} not divisible by H*W={P}. "
            f"Cannot derive per-pixel saliency."
        )

    Q = F_total // P

    # NOTE: we do NOT take abs() here; we preserve sign and only average across extra dims.
    imp_flat = imp.reshape(N, F_total)
    imp_flat = imp_flat.reshape(N, Q, P)
    imp_pix = imp_flat.mean(axis=1)      # (N,P) signed
    imp_map = imp_pix.reshape(N, H, W)   # (N,H,W)

    return imp_map

# --------------------------------------------------------------------
# Image metric evaluation (use Image* classes directly)
# --------------------------------------------------------------------
IMAGE_METRIC_CLASS = {
    "AIM": ImageAIM,
    "AUM": ImageAUM,
    "PGI": ImagePGI,
    "PGU": ImagePGU,
    "RIS": ImageRIS,
}

def evaluate_image_metric(
    metric_name: str,
    env_obj,
    X_img: np.ndarray,
    y_act: np.ndarray,
    importance_raw,
    k: int,
    explainer_obj=None,
):
    cls = IMAGE_METRIC_CLASS.get(metric_name)
    if cls is None:
        raise ValueError(f"No Image metric class mapped for '{metric_name}'.")

    metric = cls(environment=env_obj)

    if metric_name == "RIS":
        # RIS must compare feature_weights to perturbed_weights, so BOTH must be same shape.
        # We pass (N,H,W) feature_weights and wrap the explainer so that perturbed_weights
        # computed inside ImageRIS also becomes (N,H,W).
        feature_weights = prepare_importance_for_image_metrics(importance_raw, X_img)
        explainer_wrapped = ImageExplainerReducer(explainer_obj) if explainer_obj is not None else None

        try:
            return metric.evaluate(X_img, y_act, feature_weights, k=k, explainer=explainer_wrapped)
        except TypeError:
            try:
                return metric.evaluate(X_img, y_act, feature_weights, explainer=explainer_wrapped)
            except TypeError:
                return metric.evaluate(X_img, y_act, feature_weights)

    # Non-RIS image metrics: (N,H,W) is sufficient
    feature_weights = prepare_importance_for_image_metrics(importance_raw, X_img)

    try:
        return metric.evaluate(X_img, y_act, feature_weights, k=k, explainer=explainer_obj)
    except TypeError:
        pass
    try:
        return metric.evaluate(X_img, y_act, feature_weights, k=k)
    except TypeError:
        pass
    try:
        return metric.evaluate(X_img, y_act, feature_weights, explainer=explainer_obj)
    except TypeError:
        pass
    return metric.evaluate(X_img, y_act, feature_weights)



# --------------------------------------------------------------------
# Worker logic
# --------------------------------------------------------------------
def _run_one_method_worker(
    worker_id: int,
    environment_name: str,
    method: str,
    metrics: list[str],
    n: int,
    random_state: int,
    k: int,
    deletion_fractions: np.ndarray | None,
    deletion_replacement: str,
):
    log_lines: list[str] = []

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] [Worker {worker_id}] {msg}")

    try:
        method_seed = abs(hash((environment_name, method, random_state))) % (2**32)
        set_all_seeds(method_seed)
        log(f"Seed={method_seed}  Method={method}  Env={environment_name}")

        env = Environment(environment_name=environment_name)
        rows: list[dict] = []

        # ---------------- TABULAR ----------------
        if is_tabular_env(environment_name):
            log("Loading tabular dataset via Environment.get_dataset...")
            df = env.get_dataset(generate=False)
            df_sample = df.sample(n=min(n, len(df)), random_state=random_state)

            y = df_sample["action"]
            drop_cols = [c for c in ["action", "reward"] if c in df_sample.columns]
            X = df_sample.drop(columns=drop_cols)
            feature_names = list(X.columns)

            t0 = time.time()
            log("Building tabular explainer...")
            if method == "tabularShap":
                explainer = Explainer(method=method, state=X, action=y)
            else:
                explainer = Explainer(method=method, state=X, action=y, model=env.model)

            importance = explainer.explain()
            explain_time = time.time() - t0
            log(f"Explanation complete in {explain_time:.2f}s")

            for metric_name in metrics:
                if not metric_supported(environment_name, metric_name):
                    continue

                log(f"Evaluating metric {metric_name}...")
                t1 = time.time()
                evaluator = Evaluator(metric=metric_name, environment=env)

                if metric_name == "RIS":
                    result = evaluator.evaluate(X, y, importance, explainer=explainer)
                elif metric_name == "DeletionCurve":
                    baselines = ["mean", "median", "zero"] if deletion_replacement == "all" else [deletion_replacement]
                    for baseline in baselines:
                        kwargs = {"feature_names": feature_names, "baseline": baseline}
                        if deletion_fractions is not None:
                            kwargs["fractions"] = deletion_fractions
                        result = evaluator.evaluate(X, y, importance, **kwargs)

                        row = {
                            "environment": environment_name,
                            "method": method,
                            "metric": "DeletionCurve",
                            "deletion_replacement": baseline,
                            "k": k,
                            "status": "OK",
                            "explain_time_sec": explain_time,
                            "eval_time_sec": time.time() - t1,
                        }
                        if isinstance(result, dict):
                            if "auc" in result:
                                row["auc"] = _safe_float(result["auc"])
                            if "curve" in result:
                                row["curve"] = result["curve"]
                        else:
                            row["value"] = _safe_float(result)
                        rows.append(row)
                    log("DeletionCurve done.")
                    continue
                else:
                    result = evaluator.evaluate(X, y, importance, k=k)

                eval_time = time.time() - t1
                log(f"{metric_name} done in {eval_time:.2f}s")

                row = {
                    "environment": environment_name,
                    "method": method,
                    "metric": metric_name,
                    "deletion_replacement": "",
                    "k": k,
                    "status": "OK",
                    "explain_time_sec": explain_time,
                    "eval_time_sec": eval_time,
                }

                if isinstance(result, dict):
                    if "auc" in result:
                        row["auc"] = _safe_float(result["auc"])
                    if "curve" in result:
                        row["curve"] = result["curve"]
                    else:
                        for kk, vv in result.items():
                            if kk not in ("fractions",):
                                row[kk] = _safe_float(vv)
                else:
                    row["value"] = _safe_float(result)
                rows.append(row)

        # ---------------- IMAGE ----------------
        elif is_image_env(environment_name):
            log("Loading image dataset from H5...")
            X_all, y_all = load_image_dataset_for_env(environment_name, log)
            N_total = X_all.shape[0]
            rng = np.random.RandomState(random_state)

            # Ensure enough samples for background-based explainers that internally pick up to 100
            if method in {"deepShap", "gradientShap", "integratedGradient"}:
                n_expl = min(max(n, 100), N_total)
            else:
                n_expl = min(n, N_total)

            idx = rng.choice(N_total, size=n_expl, replace=False)
            X = X_all[idx]
            y = y_all[idx]
            log(f"Sampled {n_expl} images out of {N_total} for explanation & metrics")

            t0 = time.time()
            log("Building image explainer...")
            explainer = build_image_explainer(method, X, y, env.model)
            importance_raw = explainer.explain()
            explain_time = time.time() - t0
            log(f"Explanation complete in {explain_time:.2f}s")

            for metric_name in metrics:
                if not metric_supported(environment_name, metric_name):
                    continue

                t1 = time.time()

                if metric_name in IMAGE_METRIC_CLASS:
                    log(f"Evaluating image metric {metric_name} (Image{metric_name})...")
                    result = evaluate_image_metric(
                        metric_name,
                        env.environment,
                        X,
                        y,
                        importance_raw,
                        k=k,
                        explainer_obj=explainer,
                    )

                    eval_time = time.time() - t1
                    log(f"{metric_name} done in {eval_time:.2f}s")

                    row = {
                        "environment": environment_name,
                        "method": method,
                        "metric": metric_name,                 # label in CSV
                        "metric_impl": f"Image{metric_name}",  # underlying impl
                        "deletion_replacement": "",
                        "k": k,
                        "status": "OK",
                        "explain_time_sec": explain_time,
                        "eval_time_sec": eval_time,
                    }

                    if isinstance(result, dict):
                        if "auc" in result:
                            row["auc"] = _safe_float(result["auc"])
                        if "curve" in result:
                            row["curve"] = result["curve"]
                        else:
                            for kk, vv in result.items():
                                if kk not in ("fractions",):
                                    row[kk] = _safe_float(vv)
                    else:
                        row["value"] = _safe_float(result)

                    rows.append(row)
                    continue

                if metric_name == "ImageDeletionCurve":
                    log("Evaluating ImageDeletionCurve...")
                    baselines = ["mean", "median", "zero"] if deletion_replacement == "all" else [deletion_replacement]

                    # ImageDeletionCurve needs per-pixel map
                    imp_map = prepare_importance_for_image_metrics(importance_raw, X)

                    for baseline in baselines:
                        mobj = ImageDeletionCurve(
                            environment=env.environment,
                            baseline=baseline,
                            fractions=deletion_fractions,
                        )
                        result = mobj.evaluate(X, y, imp_map)

                        eval_time = time.time() - t1
                        log(f"ImageDeletionCurve({baseline}) done in {eval_time:.2f}s")

                        row = {
                            "environment": environment_name,
                            "method": method,
                            "metric": "ImageDeletionCurve",
                            "metric_impl": "ImageDeletionCurve",
                            "deletion_replacement": baseline,
                            "k": k,
                            "status": "OK",
                            "explain_time_sec": explain_time,
                            "eval_time_sec": eval_time,
                        }
                        if isinstance(result, dict):
                            if "auc" in result:
                                row["auc"] = _safe_float(result["auc"])
                            if "curve" in result:
                                row["curve"] = result["curve"]
                        else:
                            row["value"] = _safe_float(result)

                        rows.append(row)
                    continue

        else:
            raise ValueError(f"Environment '{environment_name}' is neither tabular nor image.")

        log("Completed successfully.")
        return rows, log_lines, "done"

    except Exception as e:
        log(f"ERROR: {e!r}")
        log(traceback.format_exc())
        return [], log_lines, "error"


# --------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------
@dataclass
class WorkerState:
    method: str
    status: str = "idle"  # idle | busy | done | error
    log: list[str] = field(default_factory=list)


class BenchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XRL-Bench Benchmark GUI")
        self.geometry("1100x750")

        self.environments = get_available_environments()
        self.methods_all = METHODS_ALL
        self.metrics_all = METRICS_ALL

        self.method_vars = {m: tk.BooleanVar(value=True) for m in self.methods_all}
        self.metric_vars = {m: tk.BooleanVar(value=True) for m in self.metrics_all}

        self.env_var = tk.StringVar(value=self.environments[0] if self.environments else "lunarLander")
        self.threads_var = tk.StringVar(value="6")  # default max threads = 6
        self.random_state_var = tk.StringVar(value="42")
        self.k_var = tk.StringVar(value="3")
        self.n_var = tk.StringVar(value="10")  # default n = 10 for testing
        self.del_frac_var = tk.StringVar(value="0, 1, 9")
        self.del_repl_var = tk.StringVar(value="all")

        self.output_csv_path = None
        self.run_in_progress = False

        self.worker_states: dict[int, WorkerState] = {}
        self.selected_worker_id: int | None = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Environment:").pack(side="left")
        self.env_combo = ttk.Combobox(
            top,
            textvariable=self.env_var,
            values=self.environments,
            state="readonly",
            width=30,
        )
        self.env_combo.pack(side="left", padx=8)
        self.env_combo.bind("<<ComboboxSelected>>", self.on_environment_changed)

        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.pack(fill="x")

        left = ttk.LabelFrame(mid, text="Methods", padding=10)
        right = ttk.LabelFrame(mid, text="Metrics", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        self.method_checkbuttons: dict[str, ttk.Checkbutton] = {}
        for m in self.methods_all:
            cb = ttk.Checkbutton(left, text=m, variable=self.method_vars[m])
            cb.pack(anchor="w")
            self.method_checkbuttons[m] = cb

        ttk.Label(left, text="Threads:").pack(anchor="w", pady=(10, 0))
        self.threads_combo = ttk.Combobox(
            left,
            textvariable=self.threads_var,
            values=[str(i) for i in range(1, 7)],
            state="readonly",
            width=8,
        )
        self.threads_combo.pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Random_state:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.random_state_var, width=12).pack(anchor="w")

        ttk.Label(left, text="n (samples):").pack(anchor="w", pady=(10, 0))
        ttk.Entry(left, textvariable=self.n_var, width=12).pack(anchor="w")

        self.metric_checkbuttons: dict[str, ttk.Checkbutton] = {}
        for m in self.metrics_all:
            cb = ttk.Checkbutton(right, text=m, variable=self.metric_vars[m])
            cb.pack(anchor="w")
            self.metric_checkbuttons[m] = cb

        ttk.Label(right, text="k:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(right, textvariable=self.k_var, width=12).pack(anchor="w")

        ttk.Label(right, text="Deletion fractions (start,end,num or list):").pack(anchor="w", pady=(10, 0))
        ttk.Entry(right, textvariable=self.del_frac_var, width=30).pack(anchor="w")

        ttk.Label(right, text="Replacement:").pack(anchor="w", pady=(10, 0))
        self.del_repl_combo = ttk.Combobox(
            right,
            textvariable=self.del_repl_var,
            values=["all", "mean", "median", "zero"],
            state="readonly",
            width=12,
        )
        self.del_repl_combo.pack(anchor="w")

        bottom_controls = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom_controls.pack(fill="x")

        self.start_btn = tk.Button(
            bottom_controls,
            text="Start",
            command=self.on_start_clicked,
            bg="#f0f0f0",
            fg="black",
            padx=18,
            pady=8,
        )
        self.start_btn.pack(anchor="center")

        panel = ttk.LabelFrame(self, text="Workers / Logs", padding=10)
        panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        hint = ttk.Label(
            panel,
            text="Yellow = busy, Green = done, Red = error. Select a worker to view its log.",
        )
        hint.pack(side="bottom", fill="x", pady=(8, 0))
        hint.configure(anchor="center")

        body = ttk.Frame(panel)
        body.pack(side="top", fill="both", expand=True)

        left_panel = ttk.Frame(body)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        self.tree = ttk.Treeview(
            left_panel,
            columns=("status", "method"),
            show="headings",
            height=18,
        )
        self.tree.heading("status", text="Status")
        self.tree.heading("method", text="Method")
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("method", width=200, anchor="w")
        self.tree.pack(fill="y", expand=False)

        self.tree.tag_configure("busy", foreground="#b58900")
        self.tree.tag_configure("done", foreground="#2aa198")
        self.tree.tag_configure("error", foreground="#dc322f")
        self.tree.tag_configure("idle", foreground="#586e75")

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        right_panel = ttk.Frame(body)
        right_panel.pack(side="left", fill="both", expand=True)

        self.log_text = tk.Text(right_panel, wrap="word", height=20)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        self.on_environment_changed()

    def on_environment_changed(self, event=None):
        env_name = self.env_var.get().strip()

        # Methods
        for m in self.methods_all:
            supported = method_supported(env_name, m)
            cb = self.method_checkbuttons[m]
            if supported:
                cb.configure(state="normal")

                # For image environments, auto-enable GradientShap
                if is_image_env(env_name) and m == "gradientShap":
                    self.method_vars[m].set(True)
            else:
                cb.configure(state="disabled")
                self.method_vars[m].set(False)

        # Metrics
        for m in self.metrics_all:
            supported = metric_supported(env_name, m)
            cb = self.metric_checkbuttons[m]
            if supported:
                cb.configure(state="normal")

                # For image environments, auto-enable ImageDeletionCurve
                if is_image_env(env_name) and m == "ImageDeletionCurve":
                    self.metric_vars[m].set(True)
            else:
                cb.configure(state="disabled")
                self.metric_vars[m].set(False)


    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item_id = sel[0]
        try:
            worker_id = int(item_id.replace("worker_", ""))
        except Exception:
            return
        self.selected_worker_id = worker_id
        self._refresh_log_view()

    def on_start_clicked(self):
        if (not self.run_in_progress) and self.output_csv_path and self.start_btn["text"] == "Done!":
            self._open_output_file(self.output_csv_path)
            return

        if self.run_in_progress:
            messagebox.showinfo("Benchmark running", "A run is already in progress.")
            return

        env_name = self.env_var.get().strip()
        methods = [m for m, v in self.method_vars.items() if v.get()]
        metrics = [m for m, v in self.metric_vars.items() if v.get()]

        if not env_name:
            messagebox.showerror("Input error", "Please select an environment.")
            return
        if not methods:
            messagebox.showerror("Input error", "Please select at least one method.")
            return
        if not metrics:
            messagebox.showerror("Input error", "Please select at least one metric.")
            return

        try:
            threads = int(self.threads_var.get())
            if threads < 1 or threads > 6:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "Threads must be an integer from 1 to 6.")
            return

        try:
            random_state = int(self.random_state_var.get())
        except Exception:
            messagebox.showerror("Input error", "Random_state must be an integer.")
            return

        try:
            k = int(self.k_var.get())
            if k < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "k must be a positive integer.")
            return

        try:
            n = int(self.n_var.get())
            if n < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "n must be a positive integer.")
            return

        try:
            del_fracs = parse_deletion_fractions(self.del_frac_var.get())
        except Exception as e:
            messagebox.showerror("Input error", f"Deletion fractions invalid: {e}")
            return

        del_repl = self.del_repl_var.get().strip().lower()
        if del_repl not in {"all", "mean", "median", "zero"}:
            messagebox.showerror("Input error", "Replacement must be one of: all, mean, median, zero.")
            return

        self.run_in_progress = True
        self.output_csv_path = None
        self.start_btn.configure(text="Running...", bg="#b58900", fg="black")
        self._init_workers_view(methods, threads)

        t = threading.Thread(
            target=self._run_benchmark_background,
            args=(env_name, methods, metrics, threads, random_state, k, del_fracs, del_repl, n),
            daemon=True,
        )
        t.start()

    def _init_workers_view(self, methods: list[str], threads: int):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.worker_states.clear()
        self.selected_worker_id = None
        self._set_log_text("")

        for i in range(1, threads + 1):
            self.worker_states[i] = WorkerState(method="(idle)", status="idle", log=[])
            self.tree.insert("", "end", iid=f"worker_{i}", values=("idle", "(idle)"), tags=("idle",))

        for idx, method in enumerate(methods):
            worker_id = (idx % threads) + 1
            ws = self.worker_states[worker_id]
            ws.method = method if ws.method == "(idle)" else f"{ws.method}, {method}"

        for worker_id, ws in self.worker_states.items():
            self.tree.item(f"worker_{worker_id}", values=(ws.status, ws.method), tags=(ws.status,))

        if threads >= 1:
            self.tree.selection_set("worker_1")
            self.selected_worker_id = 1
            self._refresh_log_view()

    def _set_worker_status(self, worker_id: int, status: str):
        ws = self.worker_states.get(worker_id)
        if not ws:
            return
        ws.status = status
        self.tree.item(f"worker_{worker_id}", values=(status, ws.method), tags=(status,))

    def _append_worker_log(self, worker_id: int, lines: list[str]):
        ws = self.worker_states.get(worker_id)
        if not ws:
            return
        ws.log.extend(lines)
        if self.selected_worker_id == worker_id:
            self._refresh_log_view()

    def _refresh_log_view(self):
        if self.selected_worker_id is None:
            self._set_log_text("")
            return
        ws = self.worker_states.get(self.selected_worker_id)
        if not ws:
            self._set_log_text("")
            return
        self._set_log_text("\n".join(ws.log))

    def _set_log_text(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", text)
        self.log_text.configure(state="disabled")

    def _run_benchmark_background(
        self,
        environment_name: str,
        methods: list[str],
        metrics: list[str],
        threads: int,
        random_state: int,
        k: int,
        deletion_fractions: np.ndarray,
        deletion_replacement: str,
        n: int,
    ):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = os.path.abspath(f"benchmark_results_{environment_name}_{ts}.csv")

            all_rows: list[dict] = []
            futures = []

            with ProcessPoolExecutor(max_workers=threads) as ex:
                for idx, method in enumerate(methods):
                    worker_id = (idx % threads) + 1
                    self.after(0, self._set_worker_status, worker_id, "busy")
                    self.after(0, self._append_worker_log, worker_id, [f"[GUI] Submitted method '{method}'"])

                    fut = ex.submit(
                        _run_one_method_worker,
                        worker_id,
                        environment_name,
                        method,
                        metrics,
                        n,
                        random_state,
                        k,
                        deletion_fractions,
                        deletion_replacement,
                    )
                    futures.append((fut, worker_id, method))

                future_map = {fut: (worker_id, method) for fut, worker_id, method in futures}

                for fut in as_completed(future_map.keys()):
                    worker_id, method = future_map[fut]
                    try:
                        rows, log_lines, status = fut.result()
                        all_rows.extend(rows)
                        self.after(0, self._append_worker_log, worker_id, log_lines)
                        self.after(0, self._set_worker_status, worker_id, "done" if status == "done" else "error")
                    except Exception as e:
                        tb = traceback.format_exc()
                        self.after(0, self._append_worker_log, worker_id, [f"[Worker {worker_id}] CRASH: {e!r}", tb])
                        self.after(0, self._set_worker_status, worker_id, "error")

            df = pd.DataFrame(all_rows)
            df.to_csv(output_csv, index=False)

            self.output_csv_path = output_csv
            self.run_in_progress = False

            def mark_done():
                self.start_btn.configure(text="Done!", bg="#2aa198", fg="black")
                for wid, ws in self.worker_states.items():
                    if ws.status == "busy":
                        self._set_worker_status(wid, "done")
                messagebox.showinfo("Benchmark complete", f"Saved results to:\n{output_csv}\n\nClick 'Done!' to open it.")

            self.after(0, mark_done)

        except Exception as e:
            self.run_in_progress = False

            def mark_failed():
                self.start_btn.configure(text="Start", bg="#f0f0f0", fg="black")
                messagebox.showerror("Benchmark failed", f"{e!r}")

            self.after(0, mark_failed)

    def _open_output_file(self, path: str):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # noqa: S606
            elif sys.platform.startswith("darwin"):
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as e:
            messagebox.showerror("Open file failed", f"Could not open file:\n{path}\n\nError: {e!r}")


if __name__ == "__main__":
    app = BenchGUI()
    app.mainloop()
