# -*- coding: utf-8 -*-

import shap
import torch
import numpy as np
import pandas as pd


class DeepSHAP:
    def __init__(self, X, y=None, model=None, background=None,
                 background_size=32, batch_size=16, seed=42, **kwargs):
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        rng = np.random.default_rng(seed)

        # Ensure numpy array, float32 in [0,1]
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if X_np.dtype == np.uint8:
            X_np = X_np.astype(np.float32) / 255.0
        else:
            X_np = X_np.astype(np.float32)

        # ---- pick background safely ----
        if background is None:
            N = X_np.shape[0]
            k = min(background_size, N)          # cap to available
            # if N == 0, raise; if N < k, k==N; if N==k==0, error
            if k == 0:
                raise ValueError("Empty input X: no samples available for background.")
            idx = rng.choice(N, size=k, replace=False)  # now always valid
            bg_np = X_np[idx]
        else:
            # user-provided background; normalize dtype/range
            bg_np = np.asarray(background)
            if bg_np.dtype == np.uint8:
                bg_np = bg_np.astype(np.float32) / 255.0
            else:
                bg_np = bg_np.astype(np.float32)

        self.background = torch.from_numpy(bg_np).to(self.device)

        import shap
        self.explainer = shap.DeepExplainer(self.model, self.background)

        self.batch_size = batch_size
    # def __init__(self, X, y, model, background=None):
    #     """
    #     Class for explaining the model prediction using DeepSHAP. https://arxiv.org/abs/1704.02685
    #
    #     Parameters:
    #     -----------
    #     X : pandas.DataFrame
    #         The input data for the model.
    #     y : pandas.Series or numpy.ndarray
    #         The output data for the model.
    #     model : object
    #         The trained deep model used for making predictions.
    #     background : numpy.ndarray or pandas.DataFrame, optional (default=None)
    #         The background dataset to use for integrating out features. 100-1000 samples will be good.
    #
    #     Attributes:
    #     -----------
    #     explainer : shap.DeepExplainer
    #         The SHAP explainer used for computing the SHAP values.
    #     """
    #     # Check inputs
    #     if not isinstance(X, pd.DataFrame):
    #         raise TypeError("X must be a pandas.DataFrame")
    #     if not isinstance(y, (np.ndarray, pd.Series)):
    #         raise TypeError("y must be a numpy.ndarray or pandas.Series")
    #     self.X = X
    #     self.y = y
    #     self.model = model
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     np.random.seed(42)
    #     self.background = background if background else X.values[np.random.choice(X.shape[0], 100, replace=False)]
    #     self.explainer = shap.DeepExplainer(model, torch.from_numpy(self.background).float().to(self.device))

    # def explain(self, X=None):
    #     """
    #     Explain the input data.
    #
    #     Parameters:
    #     -----------
    #     X : pandas.DataFrame, optional (default=None)
    #         The input data of shape (n_samples, n_features).
    #
    #     Returns:
    #     --------
    #     shap_values : array or list
    #         For a models with a single output this returns a tensor of SHAP values with the same shape
    #         as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
    #         which are the same shape as X.
    #     """
    #     if X is None:
    #         X = self.X
    #     X = X.values if isinstance(X, pd.DataFrame) else X
    #     shap_values = self.explainer.shap_values(torch.from_numpy(X).float().to(self.device))
    #     return np.array(shap_values).transpose((1, 2, 0))
    def explain(self, X, **kwargs):
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if X_np.dtype == np.uint8:
            X_np = X_np.astype(np.float32) / 255.0
        else:
            X_np = X_np.astype(np.float32)

        X_t = torch.from_numpy(X_np).to(self.device)

        outs = []
        for i in range(0, X_t.shape[0], self.batch_size):
            xb = X_t[i:i+self.batch_size]
            sv = self.explainer.shap_values(xb, check_additivity=False)
            if isinstance(sv, list):
                sv = sv[0]
            if torch.is_tensor(sv):
                sv = sv.detach().cpu().numpy()
            outs.append(sv)
        return np.concatenate(outs, axis=0)


class ImageDeepSHAP:
    def __init__(self, X, y, model, background=None):
        """
        Class for explaining the model prediction using DeepSHAP. https://arxiv.org/abs/1704.02685

        Parameters:
        -----------
        X : numpy.ndarray
            The input data for the model.
        y : pandas.Series or numpy.ndarray
            The output data for the model.
        model : object
            The trained deep model used for making predictions.
        background : numpy.ndarray or pandas.DataFrame, optional (default=None)
            The background dataset to use for integrating out features. 100-1000 samples will be good.

        Attributes:
        -----------
        explainer : shap.DeepExplainer
            The SHAP explainer used for computing the SHAP values.
        """
        # Check inputs
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a pandas.DataFrame")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        self.X = X
        self.y = y
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(42)

        if background is not None:
            self.background = background
        else:
            max_bg = 100
            n_bg = min(max_bg, X.shape[0])
            idx_bg = np.random.choice(X.shape[0], n_bg, replace=False)
            self.background = X[idx_bg]

        self.explainer = shap.DeepExplainer(
            model,
            torch.from_numpy(self.background).float().to(self.device)
        )

    def explain(self, X=None):
        """
        Explain the input data.

        Parameters:
        -----------
        X : numpy.ndarray, optional (default=None)
            The input data of shape  (n_samples, n_channels, n_widths, n_heights).

        Returns:
        --------
        shap_values : array or list
            For a models with a single output this returns a tensor of SHAP values with the same shape
            as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
            which are the same shape as X.
        """
        if X is None:
            X = self.X

        self.model.eval()

        x = torch.from_numpy(X).float().to(self.device)
        x.requires_grad_(True)

        with torch.enable_grad():
            shap_values = self.explainer.shap_values(
                x,
                check_additivity=False  # keep this if you were hitting the additivity assertion
            )

        return np.array(shap_values).transpose((1, 2, 3, 4, 0))