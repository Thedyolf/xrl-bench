# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
try:
    import lime
    import lime.lime_tabular
except ImportError:
    pass

def _softmax(z, axis=1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

class TabularLime:
    def __init__(self, X, y, model, categorical_names=None, mode="classification"):
        """
        Class for explaining the predictions of tarbular models using LIME. https://arxiv.org/abs/1602.04938

        Parameters:
        -----------
        X : pandas.DataFrame
            The input data of shape (n_samples, n_features).
        y : pandas.Series or numpy.ndarray
            The label of the input data of shape (n_sample,).
        model : callable or object
            The trained model used for making predictions.
        categorical_names : list, optional (default=None)
            List of categorical feature names.
        mode : str, optional (default="classification")
            The mode of the model, either "classification" or "regression".

        Attributes:
        -----------
        feature_names : list
            List of feature names.
        categorical_index : list
            List of categorical feature indices.
        explainer : lime.lime_tabular.LimeTabularExplainer
            The LIME explainer object.
        out_dim : int
            Number of output dimensions.
        flat_out : bool
            Whether the output is flat (1D) or not.

        Methods:
        --------
        explain(X=None):
            Explain the input feature data by calculating the importance scores.
        """
        # Check inputs
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame")
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be a numpy.ndarray or pandas.Series")
        self.feature_names = list(X.columns)
        self.X = X.values
        self.y = y.values if isinstance(y, pd.Series) else y
        self.model = model
        self.torch_model = model
        assert mode in ["classification", "regression"]
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.categorical_names = categorical_names if categorical_names else []
        self.categorical_index = [self.feature_names.index(state) for state in categorical_names] if categorical_names else []
        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.X, mode=mode, feature_names=self.feature_names, categorical_features=self.categorical_index)

        out = self.model(torch.from_numpy(self.X[0:1]).float().to(self.device))

        # Normalize output to (1, out_dim)
        if len(out.shape) == 1:
            out = out.reshape(1, -1)

        self.out_dim = out.shape[1]
        self.flat_out = (self.out_dim == 1)

        # If classification, ensure LIME sees probabilities
        if self.mode == "classification":
            orig_model = self.model
            device = self.device
            out_dim = self.out_dim

            def pred_proba(X_np):
                # LIME passes numpy arrays
                X_t = torch.from_numpy(np.asarray(X_np)).float().to(device)
                with torch.no_grad():
                    scores = orig_model(X_t)

                scores = scores.detach().cpu().numpy()

                # Ensure 2D
                if scores.ndim == 1:
                    scores = scores.reshape(-1, 1)

                # Binary special-case: if single logit/prob returned, make 2 columns
                if scores.shape[1] == 1:
                    p1 = 1.0 / (1.0 + np.exp(-scores))  # sigmoid
                    p0 = 1.0 - p1
                    return np.hstack([p0, p1])

                # Multiclass: softmax to probabilities
                return _softmax(scores, axis=1)

            self.predict_fn = pred_proba

    def explain(self, X=None):
        """
        Explain the input feature data by calculating the importance scores.

        Parameters:
        -----------
        X : pandas.DataFrame, optional (default=None)
            The feature data for which to generate explanations. If None, use the original feature data.

        Returns:
        --------
        importance_scores : list
            List of explanations for each output dimension.
        """
        if X is None:
            X = self.X
        self.model.to("cpu")
        X = X.values if isinstance(X, pd.DataFrame) else X
        importance_scores = [np.zeros(X.shape) for _ in range(self.out_dim)]
        for i in tqdm(range(X.shape[0])):
            x = X[i]
            exp = self.explainer.explain_instance(x, self.predict_fn, labels=range(self.out_dim), num_features=X.shape[1])
            for j in range(self.out_dim):
                for k, v in exp.local_exp[j]:
                    importance_scores[j][i, k] = v
        self.torch_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return np.array(importance_scores).transpose((1, 2, 0))