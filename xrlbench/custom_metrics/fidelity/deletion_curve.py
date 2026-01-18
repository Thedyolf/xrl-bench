import numpy as np

class DeletionCurve:
    """
    Incremental deletion curve for tabular states.

    Y-axis default: action agreement with y (dataset actions) after deleting top-ranked features.
    Returns:
      - curve: list of agreement values (len = len(fractions))
      - auc: area under curve
    """

    def __init__(self, environment, baseline="mean", fractions=None, random_state=0, **kwargs):
        self.environment = environment
        self.model = getattr(environment, "model", None)
        if self.model is None:
            raise ValueError("DeletionCurve requires environment.model to be set.")

        self.baseline = baseline
        self.fractions = fractions if fractions is not None else np.linspace(0.0, 1.0, 11)
        self.random_state = random_state

    # create baseline vector, easily adjustable for custom baseline replacement strategies such as moving average or custom values
    def _get_baseline_vector(self, X):
        if self.baseline == "mean":
            return np.mean(X, axis=0)
        if self.baseline == "median":
            return np.median(X, axis=0)
        if self.baseline == "zero":
            return np.zeros(X.shape[1], dtype=X.dtype)
        raise ValueError(f"Unknown baseline={self.baseline}")

    def _softmax(self, z, axis=1):
        z = z - np.max(z, axis=axis, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=axis, keepdims=True)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _predict_action_prob(self, X, y):
        """
        Return a 1D array in [0,1] giving P(a = y_i | X_i) for each sample i.

        This replaces using unbounded logits/Q-values directly.
        """
        y = np.asarray(y).reshape(-1)
        n = y.shape[0]
        if not np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64)

        # Best case: model already provides probabilities
        if hasattr(self.model, "predict_proba"):
            P = np.asarray(self.model.predict_proba(X))  # (n, A)
            return P[np.arange(n, dtype=np.int64), y]

        # decision_function: may be (n,A) logits/margins OR (n,) binary margin
        if hasattr(self.model, "decision_function"):
            S = np.asarray(self.model.decision_function(X))
            if S.ndim == 1:
                # Binary: interpret as margin for class 1
                p1 = self._sigmoid(S)
                # Map to chosen class probability
                return np.where(y == 1, p1, 1.0 - p1)
            P = self._softmax(S, axis=1)
            return P[np.arange(n, dtype=np.int64), y]

        # Generic callable model case (common in RL): assume it returns scores per action (Q/logits)
        if callable(self.model):
            S = np.asarray(self.model(X))
            if S.ndim == 1:
                # If it's 1D, treat as binary margin
                p1 = self._sigmoid(S)
                return np.where(y == 1, p1, 1.0 - p1)
            P = self._softmax(S, axis=1)
            return P[np.arange(n, dtype=np.int64), y]

        raise AttributeError(
            "Cannot compute action probabilities. Expected predict_proba, decision_function, "
            "or a callable returning per-action scores."
        )

    def _reduce_importance(self, feature_weights, y=None):
        if hasattr(feature_weights, "values"):
            feature_weights = feature_weights.values

        fw = np.asarray(feature_weights)

        # (d,)
        if fw.ndim == 1:
            return np.abs(fw)

        # (n, d)
        if fw.ndim == 2:
            return np.mean(np.abs(fw), axis=0)

        # (n, d, a)
        if fw.ndim == 3:
            n, d, a = fw.shape

            if y is None:
                return np.mean(np.abs(fw), axis=(0, 2))

            y = np.asarray(y).reshape(-1)
            if y.shape[0] != n:
                raise ValueError(f"y length {y.shape[0]} does not match fw n {n}")

            # integer and bounds check to catch errors
            if not np.issubdtype(y.dtype, np.integer):
                y = y.astype(np.int64)
            if y.min() < 0 or y.max() >= a:
                raise ValueError(f"Action indices out of range: min={y.min()}, max={y.max()}, num_actions={a}")

            idx = np.arange(n, dtype=np.int64)
            fw_sel = fw[idx, :, y]  # (n, d)
            return np.mean(np.abs(fw_sel), axis=0)

        raise ValueError(f"Unexpected feature_weights shape: {fw.shape}")

    # helper function for agreement scores
    def _predict_actions(self, X):
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        if hasattr(self.model, "predict_best_action"):
            return self.model.predict_best_action(X)
        if hasattr(self.model, "get_action"):
            return self.model.get_action(X)
        if callable(self.model):
            scores = self.model(X)
            scores = np.asarray(scores)
            if scores.ndim >= 2:
                return np.argmax(scores, axis=1)
            return scores
        raise AttributeError("Cannot find an action prediction method on environment.model.")

    def _predict_action_scores(self, X, y):
        """
        Return a 1D array of scores for the chosen action y_i for each sample X_i.
        """
        y = np.asarray(y).reshape(-1)
        n = y.shape[0]

        # Ensure integer action indices
        if not np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64)

        # Example branch for predict_proba
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(X)  # (n, A)
            scores = np.asarray(scores)
            return scores[np.arange(n, dtype=np.int64), y]

        # Example branch for decision_function
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                return scores
            return scores[np.arange(n, dtype=np.int64), y]

        # Example generic callable model case
        if callable(self.model):
            scores = self.model(X)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                return scores
            return scores[np.arange(n, dtype=np.int64), y]

        raise AttributeError(
            "Cannot find an action-score prediction method on environment.model. "
            "Expected predict_proba, decision_function, or a callable returning scores."
        )

    # main function to evaluate new agreement and auc scores
    def evaluate_x(self, X, y, feature_weights, **kwargs):
        feature_names = kwargs.get("feature_names", None)

        baseline = kwargs.get("baseline", None)
        if baseline is not None:
            self.baseline = baseline

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        imp = self._reduce_importance(feature_weights, y=y)
        d = imp.shape[0]
        ranks = np.argsort(-imp)

        baseline_vec = self._get_baseline_vector(X)

        curve = []
        for frac in self.fractions:
            k = int(np.round(frac * d))
            k = min(k, d)

            X_del = X.copy()
            if k > 0:
                del_idx = ranks[:k]
                X_del[:, del_idx] = baseline_vec[del_idx]

                if feature_names is not None:
                    removed = [feature_names[i] for i in del_idx]
                    print(f"[DeletionCurve] fraction={frac:.2f}, removed={removed}")

            y_hat = self._predict_actions(X_del)
            y_hat = np.asarray(y_hat).reshape(-1)

            agreement = float(np.mean(y_hat == y))
            curve.append(agreement)

        auc = float(np.trapz(curve, self.fractions))
        return {"curve": curve, "fractions": list(self.fractions), "auc": auc}

    def evaluate(self, X, y, feature_weights, **kwargs):
        feature_names = kwargs.get("feature_names", None)

        baseline = kwargs.get("baseline", None)
        if baseline is not None:
            self.baseline = baseline

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        # Ensure fractions include 0 and 1 (critical for stable normalization)
        fracs = np.asarray(self.fractions, dtype=float)
        if fracs.min() > 0.0:
            fracs = np.r_[0.0, fracs]
        if fracs.max() < 1.0:
            fracs = np.r_[fracs, 1.0]
        fracs = np.unique(fracs)  # sorted unique

        imp = self._reduce_importance(feature_weights, y=y)
        d = imp.shape[0]
        ranks = np.argsort(-imp)

        baseline_vec = self._get_baseline_vector(X)

        score_matrix = []  # list of arrays (n,), one per fraction

        for frac in fracs:
            k = int(np.round(frac * d))
            k = min(k, d)

            X_del = X.copy()
            if k > 0:
                del_idx = ranks[:k]
                X_del[:, del_idx] = baseline_vec[del_idx]

                if feature_names is not None:
                    removed = [feature_names[i] for i in del_idx]
                    print(f"[DeletionCurve] fraction={frac:.2f}, removed={removed}")

            # IMPORTANT: use bounded probability of the originally chosen action
            s_f = self._predict_action_prob(X_del, y)  # in [0,1]
            score_matrix.append(s_f)

        score_matrix = np.vstack(score_matrix)  # (T, n)

        # Per-sample normalization using explicit endpoints f=0 and f=1
        s0 = score_matrix[0, :]      # f=0
        s1 = score_matrix[-1, :]     # f=1

        denom = s0 - s1
        eps = 1e-8
        denom_safe = np.where(np.abs(denom) < eps, np.sign(denom) * eps + (denom == 0) * eps, denom)

        scores_norm = (score_matrix - s1) / denom_safe  # intended: 1 at start, 0 at end

        # Enforce deletion semantics: score must not increase as more is deleted
        # cumulative minimum along fractions, per sample
        scores_norm = np.minimum.accumulate(scores_norm, axis=0)

        # Numerical safety: bound to [0,1]
        # (This should be almost redundant after using probabilities + monotone envelope,
        #  but it prevents tiny FP drift.)
        scores_norm = np.clip(scores_norm, 0.0, 1.0)

        curve = scores_norm.mean(axis=1).tolist()
        auc = float(np.trapz(curve, fracs))

        return {"curve": curve, "fractions": list(fracs), "auc": auc}

