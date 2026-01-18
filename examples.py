# # -*- coding: utf-8 -*-
#
# from xrlbench.explainers import Explainer
# from xrlbench.evaluator import Evaluator
# from xrlbench.environments import Environment
#
#
# # Tabular数据测试
# def tabular_input_test(environment, method, metric, k=3):
#     environment = Environment(environment_name=environment)
#     df = environment.get_dataset(generate=False)
#     df_sample = df.sample(n=5000, random_state=42)
#     action_sample = df_sample['action']
#     state_sample = df_sample.drop(['action', 'reward'], axis=1)
#     if method == "tabularShap":
#         explainer = Explainer(method=method, state=state_sample, action=action_sample)
#     else:
#         explainer = Explainer(method=method, state=state_sample, action=action_sample, model=environment.model)
#     importance = explainer.explain()
#     evaluator = Evaluator(metric=metric, environment=environment)
#     if metric == "RIS":
#         performance = evaluator.evaluate(state_sample, action_sample, importance, explainer=explainer)
#     elif metric == "DeletionCurve":
#         performance = evaluator.evaluate(
#             state_sample,
#             action_sample,
#             importance,
#             k=k,
#             feature_names=list(state_sample.columns)
#         )
#     else:
#         performance = evaluator.evaluate(state_sample, action_sample, importance, k=k)
#     return performance
#
# # def _extract_arrays(ds):
# #     """
# #     Support both d3rlpy 1.x (attributes) and newer variants (methods / transitions).
# #     Returns (observations, actions).
# #     """
# #     if ds is None:
# #         return None, None
# #     # d3rlpy 1.x: direct attributes
# #     if hasattr(ds, "observations") and hasattr(ds, "actions"):
# #         return ds.observations, ds.actions
# #     # Some builds expose a transitions dict or method
# #     if hasattr(ds, "to_transitions"):
# #         tr = ds.to_transitions()
# #         return tr["observations"], tr["actions"]
# #     if hasattr(ds, "transitions"):
# #         tr = ds.transitions
# #         return tr["observations"], tr["actions"]
# #     # Fallback: try common getters if present
# #     if hasattr(ds, "get_observations") and hasattr(ds, "get_actions"):
# #         return ds.get_observations(), ds.get_actions()
# #     raise AttributeError("Could not find observations/actions in the MDPDataset object.")
#
# import os
#
# def _extract_arrays(ds, h5_path=os.path.join(".", "data", "BreakOut_dataset.h5")):
#     """
#     Return (observations, actions) from a d3rlpy MDPDataset instance
#     across versions. If attributes/methods aren't present, fall back
#     to reading the H5 directly.
#     """
#     if ds is None:
#         return None, None
#
#     # 1) d3rlpy 1.x typical attributes
#     for obs_attr, act_attr in [("observations", "actions"), ("_observations", "_actions")]:
#         if hasattr(ds, obs_attr) and hasattr(ds, act_attr):
#             obs = getattr(ds, obs_attr)
#             act = getattr(ds, act_attr)
#             if obs is not None and act is not None:
#                 return obs, act
#
#     # 2) Some builds expose transitions
#     if hasattr(ds, "to_transitions"):
#         tr = ds.to_transitions()
#         if isinstance(tr, dict) and "observations" in tr and "actions" in tr:
#             return tr["observations"], tr["actions"]
#     if hasattr(ds, "transitions"):
#         tr = ds.transitions
#         if isinstance(tr, dict) and "observations" in tr and "actions" in tr:
#             return tr["observations"], tr["actions"]
#
#     # 3) Getter methods (rare)
#     if hasattr(ds, "get_observations") and hasattr(ds, "get_actions"):
#         return ds.get_observations(), ds.get_actions()
#
#     # 4) Fallback: read the H5 file directly (guaranteed keys from your dump)
#     try:
#         import h5py
#         with h5py.File(h5_path, "r") as f:
#             obs = f["observations"][:]   # (N, C, H, W) uint8
#             act = f["actions"][:]        # (N, 1) int
#         return obs, act
#     except Exception as e:
#         raise AttributeError(
#             f"Could not find observations/actions in the MDPDataset and failed to load from H5 at {h5_path}: {e}"
#         )
#
#
# # Image数据测试
# # def image_input_test(environment, method, metric, k=50):
# #     environment = Environment(environment_name=environment)
# #     dataset = environment.get_dataset(generate=False, data_format="h5")
# #     if dataset is None:
# #         dataset = environment.get_dataset(generate=True, n_episodes=10, max_t=500, data_format="h5")
# #         if dataset is None:
# #             raise RuntimeError("Dataset generation failed.")
# #     observations, actions = _extract_arrays(dataset)
# #     explainer = Explainer(method=method, state=observations, action=actions,
# #                           model=environment.model)
# #     importance = explainer.explain()
# #     evaluator = Evaluator(metric=metric, environment=environment)
# #     if metric == "RIS":
# #         performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, explainer=explainer)
# #     else:
# #         performance = evaluator.evaluate(dataset.observations, dataset.actions, importance, k=k)
# #     return performance
#
# def image_input_test(environment, method, metric, k=50):
#     environment = Environment(environment_name=environment)
#
#     dataset = environment.get_dataset(generate=False, data_format="h5")
#     if dataset is None:
#         dataset = environment.get_dataset(generate=True, n_episodes=10, max_t=500, data_format="h5")
#         if dataset is None:
#             raise RuntimeError("Dataset generation failed.")
#
#     observations, actions = _extract_arrays(dataset)
#     # N_EXPLAIN = 64
#     # observations = observations[:N_EXPLAIN]
#     # actions = actions[:N_EXPLAIN]
#     explainer = Explainer(method=method, state=observations, action=actions,
#                           model=environment.model)
#     importance = explainer.explain()
#     evaluator = Evaluator(metric=metric, environment=environment)
#
#     if metric == "RIS":
#         return evaluator.evaluate(observations, actions, importance, explainer=explainer)
#     else:
#         return evaluator.evaluate(observations, actions, importance, k=k)
#
#
#
# if __name__ == "__main__":
#     # Tabular数据测试
#     environment = "lunarLander"
#     method = "tabularShap"
#     metric = "DeletionCurve"
#
#     performance = tabular_input_test(environment, method, metric, k=3)
#     print("Deletion curve result:", performance)
#
#     # Image数据测试
#     # environment = "breakOut"
#     # method = "imageDeepShap"
#     # metric = "imageAIM"
#     # performance = image_input_test(environment, method, metric, k=50)
#
#