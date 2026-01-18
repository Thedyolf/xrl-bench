import os
import sys
import time
import queue
import random
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

import torch

from xrlbench.explainers import Explainer
from xrlbench.evaluator import Evaluator
from xrlbench.environments import Environment
import warnings

warnings.filterwarnings("ignore", message="Gym has been unmaintained since 2022.*")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message="WARN: The environment .* is out of date.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names", category=UserWarning)

# Overwrite all random seeds to same value, repeated on multithreading to set for each thread
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Helper function to cast to float
def _safe_float(x):
    if isinstance(x, (np.floating, float, int)):
        return float(x)
    return x


def parse_deletion_fractions(spec: str) -> np.ndarray:
    """
    Accepts "0, 1, 9" meaning linspace(0,1,9)
    Also accepts explicit list like "0,0.1,0.2,0.5,1"
    This allows for variable spacing between intervals, important for bigger more complex environments.
    Lots of features can be removed as multiple or singular at a time, whilst also prioritizing early accuracy.
    """
    parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
    if len(parts) < 2:
        raise ValueError("Deletion fractions must be 'start,end,num' or an explicit list.")

    nums = [float(p) for p in parts]

    # If 3 values: interpret as linspace(start, end, num)
    if len(nums) == 3:
        start, end, n = nums
        n_int = int(round(n))
        if n_int <= 1:
            raise ValueError("For 'start,end,num', num must be >= 2.")
        return np.linspace(start, end, n_int)

    # Explicit list
    return np.array(nums, dtype=float)


def get_available_environments() -> list[str]:
    """
    Auto Load environments discovered in xrl.environments
    """
    try:
        import xrlbench.environments as envmod

        for attr in ["valid_environments", "VALID_ENVIRONMENTS", "environments", "ENVIRONMENTS", "valid_envs"]:
            if hasattr(envmod, attr):
                obj = getattr(envmod, attr)
                if isinstance(obj, dict):
                    return sorted(list(obj.keys()))
                if isinstance(obj, (list, tuple, set)):
                    return sorted(list(obj))

        for name, obj in vars(envmod).items():
            if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
                keys = list(obj.keys())
                if any("lander" in k.lower() or "break" in k.lower() or "cart" in k.lower() for k in keys):
                    return sorted(keys)

    except Exception:
        pass

    # If invalid, hardcoded list:
    return [
        "lunarLander",
        "cartPole"
    ]


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
    """
    MultiThreading pet method, looping all selected metrics.
    Creates faster processing, depending on user max threads selected.
    """
    log_lines = []

    # Logging
    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")

    try:
        # Set seeds for each individual thread
        method_seed = abs(hash((environment_name, method, random_state))) % (2**32)
        set_all_seeds(method_seed)
        log(f"[Worker {worker_id}] Seed={method_seed}  Method={method}  Env={environment_name}")

        env = Environment(environment_name=environment_name)

        # Attempt to set environment seeds if supported
        try:
            if hasattr(env, "env") and hasattr(env.env, "seed"):
                env.env.seed(method_seed)
                log(f"[Worker {worker_id}] Seeded underlying env.")
        except Exception:
            pass

        df = env.get_dataset(generate=False)
        df_sample = df.sample(n=n, random_state=random_state)
        y = df_sample["action"]
        drop_cols = [c for c in ["action", "reward"] if c in df_sample.columns]
        X = df_sample.drop(columns=drop_cols)
        feature_names = list(X.columns)

        # Build explainer
        t0 = time.time()
        log(f"[Worker {worker_id}] Building explainer...")
        if method == "tabularShap":
            explainer = Explainer(method=method, state=X, action=y)
        else:
            explainer = Explainer(method=method, state=X, action=y, model=env.model)

        importance = explainer.explain()
        explain_time = time.time() - t0
        log(f"[Worker {worker_id}] Explanation complete in {explain_time:.2f}s")

        rows = []

        # Evaluate metrics
        for metric in metrics:
            log(f"[Worker {worker_id}] Evaluating metric {metric}...")
            t1 = time.time()
            evaluator = Evaluator(metric=metric, environment=env)

            if metric == "RIS":
                result = evaluator.evaluate(X, y, importance, explainer=explainer)
            elif metric == "DeletionCurve":
                # Determine which baselines to run, GUI doesn't support custom baselines replacement strategies
                if deletion_replacement == "all":
                    baselines = ["mean", "median", "zero"]
                else:
                    baselines = [deletion_replacement]

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

                continue
            else:
                result = evaluator.evaluate(X, y, importance, k=k)

            eval_time = time.time() - t1
            log(f"[Worker {worker_id}] {metric} done in {eval_time:.2f}s")

            row = {
                "environment": environment_name,
                "method": method,
                "metric": metric,
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

        log(f"[Worker {worker_id}] Completed successfully.")
        return rows, log_lines, "done"

    except Exception as e:
        log(f"[Worker {worker_id}] ERROR: {e!r}")
        log(traceback.format_exc())
        return [], log_lines, "error"


# -------------------------
# GUI Junk
# -------------------------

@dataclass
class WorkerState:
    method: str
    status: str = "idle"   # idle | busy | done | error
    log: list[str] = field(default_factory=list)


class BenchGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("XRL-Bench Tabular Benchmark GUI")
        self.geometry("1100x750")

        self.environments = get_available_environments()

        self.methods_all = ["tabularShap", "tabularLime", "sarfa", "perturbationSaliency", "deepShap", "integratedGradient"]
        self.metrics_all = ["AIM", "AUM", "PGI", "PGU", "RIS", "DeletionCurve"]

        self.method_vars = {m: tk.BooleanVar(value=True) for m in self.methods_all}
        self.metric_vars = {m: tk.BooleanVar(value=True) for m in self.metrics_all}

        self.env_var = tk.StringVar(value=self.environments[0] if self.environments else "lunarLander")
        self.threads_var = tk.StringVar(value="6")
        self.random_state_var = tk.StringVar(value="42")
        self.k_var = tk.StringVar(value="3")
        self.del_frac_var = tk.StringVar(value="0, 1, 9")
        self.del_repl_var = tk.StringVar(value="all")  # all | mean | median | zero

        self.output_csv_path = None
        self.run_in_progress = False

        # Worker panel state
        self.worker_states: dict[int, WorkerState] = {}
        self.selected_worker_id: int | None = None

        self._build_ui()

    def _build_ui(self):
        # Top row: environment selector
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Environment:").pack(side="left")
        self.env_combo = ttk.Combobox(top, textvariable=self.env_var, values=self.environments, state="readonly", width=30)
        self.env_combo.pack(side="left", padx=8)

        # Two columns
        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.pack(fill="x")

        left = ttk.LabelFrame(mid, text="Methods", padding=10)
        right = ttk.LabelFrame(mid, text="Metrics", padding=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        # Methods checkboxes
        for m in self.methods_all:
            ttk.Checkbutton(left, text=m, variable=self.method_vars[m]).pack(anchor="w")

        # Threads selection
        ttk.Label(left, text="Threads:").pack(anchor="w", pady=(10, 0))
        self.threads_combo = ttk.Combobox(left, textvariable=self.threads_var, values=[str(i) for i in range(1, 7)],
                                          state="readonly", width=8)
        self.threads_combo.pack(anchor="w", pady=(0, 6))

        # Random state
        ttk.Label(left, text="Random_state:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.random_state_var, width=12).pack(anchor="w")

        # Metrics checkboxes
        for m in self.metrics_all:
            ttk.Checkbutton(right, text=m, variable=self.metric_vars[m]).pack(anchor="w")

        # k
        ttk.Label(right, text="k:").pack(anchor="w", pady=(10, 0))
        ttk.Entry(right, textvariable=self.k_var, width=12).pack(anchor="w")

        # deletion fractions
        ttk.Label(right, text="Deletion fractions (start,end,num or list):").pack(anchor="w", pady=(10, 0))
        ttk.Entry(right, textvariable=self.del_frac_var, width=30).pack(anchor="w")

        ttk.Label(right, text="DeletionCurve replacement:").pack(anchor="w", pady=(10, 0))
        self.del_repl_combo = ttk.Combobox(
            right,
            textvariable=self.del_repl_var,
            values=["all", "mean", "median", "zero"],
            state="readonly",
            width=12,
        )
        self.del_repl_combo.pack(anchor="w")

        # Start button
        bottom_controls = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom_controls.pack(fill="x")

        self.start_btn = tk.Button(
            bottom_controls,
            text="Start",
            command=self.on_start_clicked,
            bg="#f0f0f0",
            fg="black",
            padx=18,
            pady=8
        )
        self.start_btn.pack(anchor="center")

        # Worker display area
        panel = ttk.LabelFrame(self, text="Workers / Logs", padding=10)
        panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Hint line
        hint = ttk.Label(panel, text="Yellow = busy, Green = done, Red = error. Select a worker to view its log.")
        hint.pack(side="bottom", fill="x", pady=(8, 0))
        hint.configure(anchor="center")

        # Body frame holds left+right panes
        body = ttk.Frame(panel)
        body.pack(side="top", fill="both", expand=True)

        # Left: folder-style tree
        left_panel = ttk.Frame(body)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        self.tree = ttk.Treeview(left_panel, columns=("status", "method"), show="headings", height=18)
        self.tree.heading("status", text="Status")
        self.tree.heading("method", text="Method")
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("method", width=200, anchor="w")
        self.tree.pack(fill="y", expand=False)

        # Tag colors
        self.tree.tag_configure("busy", foreground="#b58900")  # yellow-ish
        self.tree.tag_configure("done", foreground="#2aa198")  # green-ish
        self.tree.tag_configure("error", foreground="#dc322f")  # red-ish
        self.tree.tag_configure("idle", foreground="#586e75")  # gray-ish

        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Right: log viewer
        right_panel = ttk.Frame(body)
        right_panel.pack(side="left", fill="both", expand=True)

        self.log_text = tk.Text(right_panel, wrap="word", height=20)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    # -------------------------
    # UI event handlers
    # -------------------------

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
        # If completed -> open output file
        if (not self.run_in_progress) and self.output_csv_path and self.start_btn["text"] == "Done!":
            self._open_output_file(self.output_csv_path)
            return

        if self.run_in_progress:
            messagebox.showinfo("Benchmark running", "A run is already in progress.")
            return

        # Validate inputs
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
            del_fracs = parse_deletion_fractions(self.del_frac_var.get())
        except Exception as e:
            messagebox.showerror("Input error", f"Deletion fractions invalid: {e}")
            return

        del_repl = self.del_repl_var.get().strip().lower()
        if del_repl not in {"all", "mean", "median", "zero"}:
            messagebox.showerror("Input error", "DeletionCurve replacement must be one of: all, mean, median, zero.")
            return

        # Prepare UI state
        self.run_in_progress = True
        self.output_csv_path = None
        self.start_btn.configure(text="Running...", bg="#b58900", fg="black")  # yellow
        self._init_workers_view(methods, threads)

        # Run in a background thread so UI stays responsive
        t = threading.Thread(
            target=self._run_benchmark_background,
            args=(env_name, methods, metrics, threads, random_state, k, del_fracs, del_repl),
            daemon=True
        )
        t.start()

    # -------------------------
    # Worker display helpers
    # -------------------------

    def _init_workers_view(self, methods: list[str], threads: int):
        # Clear current tree/log
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.worker_states.clear()
        self.selected_worker_id = None
        self._set_log_text("")

        # Create worker rows
        for i in range(1, threads + 1):
            self.worker_states[i] = WorkerState(method="(idle)", status="idle", log=[])
            self.tree.insert("", "end", iid=f"worker_{i}", values=("idle", "(idle)"), tags=("idle",))

        # Pre-assign methods to workers in a round-robin display sense
        # This is purely for the UI
        # Reassignment of threads happen when threads finish, to ensure fastest processing
        for idx, method in enumerate(methods):
            worker_id = (idx % threads) + 1
            ws = self.worker_states[worker_id]
            if ws.method == "(idle)":
                ws.method = method
            else:
                ws.method = f"{ws.method}, {method}"

        # Update displayed method column for each worker
        for worker_id, ws in self.worker_states.items():
            self.tree.item(f"worker_{worker_id}", values=(ws.status, ws.method), tags=(ws.status,))

        # Select Worker 1 by default
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

    # -------------------------
    # Benchmark background run
    # -------------------------

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
    ):
        """
        Runs methods in parallel (process pool) and updates GUI safely via `after(...)`.
        """
        try:
            # Use timestamped output
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv = os.path.abspath(f"tabular_benchmark_results_{environment_name}_{ts}.csv")

            n = 5000
            all_rows = []
            futures = []

            with ProcessPoolExecutor(max_workers=threads) as ex:
                # Submit each method as its own job
                for idx, method in enumerate(methods):
                    worker_id = (idx % threads) + 1  # UI slot to associate with this method job
                    # Mark busy
                    self.after(0, self._set_worker_status, worker_id, "busy")
                    self.after(0, self._append_worker_log, worker_id, [f"[GUI] Submitted method '{method}'"])

                    fut = ex.submit(
                        _run_one_method_worker,
                        worker_id, environment_name, method, metrics, n, random_state, k, deletion_fractions, deletion_replacement
                    )
                    futures.append((fut, worker_id, method))

                # Collect as they finish
                for fut, worker_id, method in futures:
                    pass  # placeholders to loop below with as_completed

                future_map = {fut: (worker_id, method) for fut, worker_id, method in futures}

                for fut in as_completed(future_map.keys()):
                    worker_id, method = future_map[fut]
                    try:
                        rows, log_lines, status = fut.result()
                        all_rows.extend(rows)

                        # Update worker status & logs
                        self.after(0, self._append_worker_log, worker_id, log_lines)
                        self.after(0, self._set_worker_status, worker_id, "done" if status == "done" else "error")

                    except Exception as e:
                        tb = traceback.format_exc()
                        self.after(0, self._append_worker_log, worker_id, [f"[Worker {worker_id}] CRASH: {e!r}", tb])
                        self.after(0, self._set_worker_status, worker_id, "error")

            # Save CSV
            df = pd.DataFrame(all_rows)
            df.to_csv(output_csv, index=False)

            # Update UI done state
            self.output_csv_path = output_csv
            self.run_in_progress = False

            def mark_done():
                self.start_btn.configure(text="Done!", bg="#2aa198", fg="black")  # green
                # Also mark any still-busy workers as done (should not occur, but safe)
                for wid, ws in self.worker_states.items():
                    if ws.status == "busy":
                        self._set_worker_status(wid, "done")
                messagebox.showinfo("Benchmark complete", f"Saved results to:\n{output_csv}\n\nClick 'Done!' to open the file.")

            self.after(0, mark_done)

        except Exception as e:
            self.run_in_progress = False

            def mark_failed():
                self.start_btn.configure(text="Start", bg="#f0f0f0", fg="black")
                messagebox.showerror("Benchmark failed", f"{e!r}")

            self.after(0, mark_failed)

    # -------------------------
    # Open file helper
    # -------------------------

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
