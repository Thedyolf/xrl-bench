import numpy as np
import h5py

from xrlbench.environments import Environment
from xrlbench.custom_metrics.fidelity.image_deletion_curve import ImageDeletionCurve


def load_pong_arrays_from_h5(path: str):
    """
    Load Pong_dataset.h5 as (X, y) arrays.

    You MUST adjust obs_key / act_key if your file uses other names.
    """
    with h5py.File(path, "r") as f:
        # TODO: adjust based on inspection output if needed
        obs_key = "observations"
        act_key = "actions"

        if obs_key not in f.keys() or act_key not in f.keys():
            raise KeyError(
                f"Cannot find '{obs_key}'/'{act_key}' in {path}. "
                f"Available keys: {list(f.keys())}"
            )

        X = f[obs_key][()]      # (N,C,H,W) or (N,H,W,C)
        y = f[act_key][()].squeeze()

    return X, y


if __name__ == "__main__":
    # Environment just to keep consistent with XRL-Bench
    env = Environment("pong")

    # 1) Load raw arrays from your HDF5 file
    X_all, y_all = load_pong_arrays_from_h5("./data/Pong_dataset.h5")

    print("Loaded Pong from H5:")
    print("  X_all shape:", X_all.shape)
    print("  y_all shape:", y_all.shape)

    # 2) Take a small subset for testing
    N = 64
    X_images = X_all[:N]
    y_actions = y_all[:N]

    # 3) Dummy importance map just to test ImageDeletionCurve wiring
    #    Replace with real explainer output later.
    #    We assume X_images is (N, C, H, W); adapt if it's (N, H, W, C).
    if X_images.ndim != 4:
        raise ValueError(f"Expected X_images to be 4D (N,C,H,W) or (N,H,W,C), got shape {X_images.shape}")

    if X_images.shape[1] in (1, 3, 4):
        # Likely (N, C, H, W)
        _, C, H, W = X_images.shape
    else:
        # Likely (N, H, W, C)
        _, H, W, C = X_images.shape

    A = int(np.max(y_actions)) + 1  # number of actions (rough heuristic)

    # Dummy attributions: shape (N, H, W, C_attr, A)
    importance_map = np.random.randn(N, H, W, 1, A).astype(np.float32)

    # 4) Instantiate your metric
    metric = ImageDeletionCurve(
        environment=env,            # or environment="pong" if your class expects a name
        baseline="mean",
        fractions=np.linspace(0, 1, 11),
    )

    # 5) Evaluate
    result = metric.evaluate(
        X_images,
        y_actions,
        importance_map,
    )

    print("Curve:", result["curve"])
    print("AUC:", result["auc"])
