# -*- coding: utf-8 -*-

import gymnasium as gym
import torch
import os
import numpy as np
import torchvision.transforms as T
from collections import deque
from d3rlpy.dataset import MDPDataset
from xrlbench.custom_environment.breakout.agent import Agent
from gymnasium.envs.registration import register_envs
import ale_py

def preprocess_state(state):
    return T.Compose([T.ToPILImage(), T.Resize((84, 84)), T.ToTensor()])(state).unsqueeze(0)


class BreakOut:
    def __init__(self, env_id="ALE/Breakout-v5", render: bool = False, seed=None):
        """
        Class for constructing a BreakOut environment.
        """
        render_mode = "human" if render else None
        register_envs(ale_py)
        self.env = gym.make(env_id, render_mode=render_mode)
        self.seed = seed
        if self.seed is not None:
            # set seed at first reset call, Gymnasium recommends seeding via reset(...)
            pass

        self.agent = Agent(action_size=self.env.action_space.n)
        self.model = self.agent.policy_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categorical_states = None
        self.load_model()

    def load_model(self):
        try:
            self.model.load_state_dict(
                torch.load(os.path.join(".", "model", "BreakOut.pth"), map_location=self.device)
            )
        except Exception:
            print("This model is not existing, please train it.")

    def train_model(self, n_episodes=50000, max_t=10000, ending_score=16):
        scores_window = deque(maxlen=100)
        for i in range(1, n_episodes + 1):
            train = len(self.agent.replay_buffer) > 5000
            score = 0.0

            obs, _info = self.env.reset(seed=self.seed)
            state = preprocess_state(obs)

            for t in range(max_t):
                action = self.agent.act(np.array(state), train)
                obs_next, reward, terminated, truncated, _info = self.env.step(action)
                done = terminated or truncated

                next_state = preprocess_state(obs_next)
                self.agent.replay_buffer.add(state, action, reward, next_state, done)
                self.agent.t_step += 1

                if self.agent.t_step % self.agent.policy_update == 0:
                    self.agent.optimize_model(train)

                if self.agent.t_step % self.agent.target_update == 0:
                    self.agent.target_network.load_state_dict(self.agent.policy_network.state_dict())

                state = next_state
                score += reward
                if done:
                    print("score:", score, "t:", t)
                    break

            scores_window.append(score)
            print(f'\rEpisode {i}\tAverage Score: {np.mean(scores_window):.2f}', end='')
            if i % 100 == 0:
                print(f'\rEpisode {i}\tAverage Score: {np.mean(scores_window):.2f}')
            if np.mean(scores_window) >= ending_score:
                print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(i, np.mean(scores_window)))
                break

        torch.save(self.agent.policy_network.state_dict(), os.path.join(".", "model", "BreakOut.pth"))
        return self.agent.policy_network

    def get_dataset(self, generate=False, n_episodes=10, max_t=500, data_format="h5"):
        """
        BreakOut dataset (robust across d3rlpy versions):
        - On generate=True: create arrays, write our own flat H5, and return MDPDataset.
        - On generate=False: read our flat H5, or (if not flat) reconstruct from episode-split H5.
        """
        import h5py
        from d3rlpy.dataset import MDPDataset

        h5_path = os.path.join(".", "data", "BreakOut_dataset.h5")

        def _to_mdp(obs, acts, rews, terms):
            # Ensure shapes/dtypes that d3rlpy likes
            obs = obs.astype(np.uint8)  # (N, C, H, W)
            acts = acts.astype(np.int64)  # (N, 1)
            rews = rews.astype(np.float32)  # (N, 1)
            terms = terms.astype(np.bool_)  # (N, 1)
            return MDPDataset(obs, acts, rews, terms)

        def _load_flat_h5(path):
            with h5py.File(path, "r") as f:
                if all(k in f for k in ("observations", "actions", "rewards", "terminals")):
                    return (
                        f["observations"][:],
                        f["actions"][:],
                        f["rewards"][:],
                        f["terminals"][:],
                    )
                return None  # not our flat format

        def _load_episode_split_h5(path):
            """
            Reconstruct from per-episode layout (observations_0, actions_0, ..., terminated_0).
            Handles both Group and Dataset, and scalar-like datasets (use [()] when needed).
            """
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                # collect episode indices present
                epi_ids = sorted(
                    {int(k.split("_")[1]) for k in keys if k.startswith("observations_")}
                )
                if not epi_ids:
                    return None

                def _read_any(node):
                    """
                    Read an HDF5 node that may be a Dataset (array or scalar) or a Group
                    containing a dataset. Uses node[()] for 0-D (scalar) datasets.
                    """
                    import h5py as _h5

                    if isinstance(node, _h5.Dataset):
                        # 0-D (scalar) -> use [()], arrays -> use [...]
                        return node[()] if node.shape == () else node[...]
                    elif isinstance(node, _h5.Group):
                        # Prefer a direct dataset inside this group
                        for sub in node.values():
                            if isinstance(sub, _h5.Dataset):
                                return sub[()] if sub.shape == () else sub[...]
                        # If there are subgroups, recurse to find the first dataset
                        for sub in node.values():
                            if isinstance(sub, _h5.Group):
                                val = _read_any(sub)
                                if val is not None:
                                    return val
                        raise KeyError("Group contains no dataset")
                    else:
                        raise TypeError(f"Unknown H5 node type: {type(node)}")

                obs_chunks, act_chunks, rew_chunks, term_chunks = [], [], [], []
                for i in epi_ids:
                    okey, akey, rkey, tkey = (
                        f"observations_{i}",
                        f"actions_{i}",
                        f"rewards_{i}",
                        f"terminated_{i}",
                    )
                    if okey not in f or akey not in f or rkey not in f or tkey not in f:
                        # skip incomplete episode
                        continue
                    obs_chunks.append(_read_any(f[okey]))
                    act_chunks.append(_read_any(f[akey]))
                    rew_chunks.append(_read_any(f[rkey]))
                    term_chunks.append(_read_any(f[tkey]))

                if not obs_chunks:
                    return None

                observations = np.concatenate(obs_chunks, axis=0)  # (N, C, H, W)
                actions = np.concatenate(act_chunks, axis=0)
                rewards = np.concatenate(rew_chunks, axis=0)
                terminals = np.concatenate(term_chunks, axis=0)

                # Ensure (N,1) for non-image arrays
                if actions.ndim == 1:   actions = actions.reshape(-1, 1)
                if rewards.ndim == 1:   rewards = rewards.reshape(-1, 1)
                if terminals.ndim == 1: terminals = terminals.reshape(-1, 1)

                return observations, actions, rewards, terminals

        if generate:
            data = []
            for _ in range(n_episodes):
                obs, _info = self.env.reset(seed=self.seed)
                state = preprocess_state(obs)  # (1, C, H, W)

                for _t in range(max_t):
                    action = self.agent.act(np.array(state), inferring=True)
                    obs_next, reward, terminated, truncated, _info = self.env.step(action)
                    done = terminated or truncated

                    next_state = preprocess_state(obs_next)

                    state_np = state.squeeze(0).cpu().numpy()  # (C,H,W) float [0,1]
                    data.append({
                        "state": (state_np * 255).astype(np.uint8),  # (C,H,W)
                        "action": np.array([action], dtype=np.int64),  # (1,)
                        "reward": np.array([reward], dtype=np.float32),  # (1,)
                        "terminal": np.array([done], dtype=np.bool_),  # (1,)
                    })

                    state = next_state
                    if done:
                        break

            # Build flat arrays
            observations = np.stack([r["state"] for r in data], axis=0)  # (N,C,H,W)
            actions = np.vstack([r["action"] for r in data])  # (N,1)
            rewards = np.vstack([r["reward"] for r in data])  # (N,1)
            terminals = np.vstack([r["terminal"] for r in data])  # (N,1)

            # Write our own flat H5
            os.makedirs(os.path.dirname(h5_path), exist_ok=True)
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("observations", data=observations, compression="gzip")
                f.create_dataset("actions", data=actions, compression="gzip")
                f.create_dataset("rewards", data=rewards, compression="gzip")
                f.create_dataset("terminals", data=terminals, compression="gzip")

            # Return an MDPDataset instance for the caller
            return _to_mdp(observations, actions, rewards, terminals)

        # -------- load path --------
        if not os.path.exists(h5_path):
            print("This dataset is not existing, please generate it.")
            return None

        flat = _load_flat_h5(h5_path)
        if flat is not None:
            o, a, r, t = flat
            return _to_mdp(o, a, r, t)

        epi = _load_episode_split_h5(h5_path)
        if epi is not None:
            o, a, r, t = epi
            return _to_mdp(o, a, r, t)

        print("Failed to load dataset — try regenerating.")
        return None

    #
    # def get_dataset(self, generate=False, n_episodes=10, max_t=500, data_format="h5"):
    #     """
    #     Get or generate the BreakOut dataset (compatible with all d3rlpy versions).
    #     """
    #     import importlib
    #     import inspect
    #     h5_path = os.path.join(".", "data", "BreakOut_dataset.h5")
    #
    #     def _load_dataset(path):
    #         """
    #         Compatible loader for d3rlpy v2.x (episode-split H5 files) and older versions.
    #         """
    #         from d3rlpy.dataset import MDPDataset
    #         import h5py, numpy as np, inspect
    #
    #         # --- Try legacy MDPDataset.load() ---
    #         if hasattr(MDPDataset, "load") and inspect.ismethod(MDPDataset.load):
    #             try:
    #                 return MDPDataset.load(path)
    #             except Exception as e:
    #                 print("Old-style MDPDataset.load failed:", e)
    #
    #         # --- Manual reconstruction ---
    #         try:
    #             with h5py.File(path, "r") as f:
    #                 keys = list(f.keys())
    #                 print("H5 keys found:", keys[:10], "..." if len(keys) > 10 else "")
    #
    #                 # Collect per-episode arrays
    #                 obs_list, act_list, rew_list, term_list = [], [], [], []
    #
    #                 for k in keys:
    #                     if k.startswith("observations_"):
    #                         idx = k.split("_")[1]
    #                         obs_list.append(f[f"observations_{idx}"][:])
    #                         act_list.append(f[f"actions_{idx}"][:])
    #                         rew_list.append(f[f"rewards_{idx}"][:])
    #                         term_list.append(f[f"terminated_{idx}"][:])
    #
    #                 if not obs_list:
    #                     raise KeyError("No observations_# groups found in H5 file.")
    #
    #                 observations = np.concatenate(obs_list, axis=0)
    #                 actions = np.concatenate(act_list, axis=0)
    #                 rewards = np.concatenate(rew_list, axis=0)
    #                 terminals = np.concatenate(term_list, axis=0)
    #
    #                 print(f"Loaded {len(observations)} transitions from {len(obs_list)} episodes.")
    #
    #             return MDPDataset(observations, actions, rewards, terminals)
    #
    #         except Exception as e:
    #             print("Could not reconstruct MDPDataset from per-episode H5 structure:", e)
    #             return None
    #
    #     # --------------------------------------------------------------------------
    #     if generate:
    #         data = []
    #         for _ in range(n_episodes):
    #             obs, _info = self.env.reset(seed=self.seed)
    #             state = preprocess_state(obs)
    #
    #             for _t in range(max_t):
    #                 action = self.agent.act(np.array(state), inferring=True)
    #                 obs_next, reward, terminated, truncated, _info = self.env.step(action)
    #                 done = terminated or truncated
    #
    #                 next_state = preprocess_state(obs_next)
    #
    #                 # normalize shapes
    #                 state_np = state.squeeze(0).cpu().numpy()
    #                 next_np = next_state.squeeze(0).cpu().numpy()
    #
    #                 data.append({
    #                     "state": (state_np * 255).astype(np.uint8),
    #                     "action": np.array([action], dtype=np.int64),
    #                     "reward": np.array([reward], dtype=np.float32),
    #                     "terminal": np.array([done], dtype=np.bool_),
    #                 })
    #
    #                 state = next_state
    #                 if done:
    #                     break
    #
    #         if data_format == "h5":
    #             observations = np.stack([r["state"] for r in data], axis=0)
    #             actions = np.vstack([r["action"] for r in data])
    #             rewards = np.vstack([r["reward"] for r in data])
    #             terminals = np.vstack([r["terminal"] for r in data])
    #
    #             from d3rlpy.dataset import MDPDataset
    #             dataset = MDPDataset(observations, actions, rewards, terminals)
    #
    #             os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    #             dataset.dump(h5_path)
    #
    #             return _load_dataset(h5_path)
    #         else:
    #             raise NotImplementedError("This data format is not supported at the moment.")
    #
    #     else:
    #         if os.path.exists(h5_path):
    #             ds = _load_dataset(h5_path)
    #             if ds is None:
    #                 print("Failed to load dataset — try regenerating.")
    #             return ds
    #         else:
    #             print("This dataset is not existing, please generate it.")
    #             return None

    # def get_dataset(self, generate=False, n_episodes=10, max_t=500, data_format="h5"):
    #     """
    #     Get the dataset for the Break Out environment.
    #     """
    #     if generate:
    #         data = []
    #         for _ in range(n_episodes):
    #             obs, _info = self.env.reset(seed=self.seed)
    #             state = preprocess_state(obs)
    #
    #             for _t in range(max_t):
    #                 action = self.agent.act(np.array(state), inferring=True)
    #                 obs_next, reward, terminated, truncated, _info = self.env.step(action)
    #                 done = terminated or truncated
    #
    #                 next_state = preprocess_state(obs_next)
    #
    #                 # store uint8 images like before
    #                 data.append({
    #                     "state": np.array(state * 255, dtype=np.uint8),
    #                     "action": np.array([action]),
    #                     "reward": np.array([reward]),
    #                     "terminal": np.array([done]),
    #                 })
    #
    #                 state = next_state
    #                 if done:
    #                     break
    #
    #         if data_format == "h5":
    #             observations = np.vstack([row["state"] for row in data])
    #             actions = np.vstack([row["action"] for row in data])
    #             rewards = np.vstack([row["reward"] for row in data])
    #             terminals = np.vstack([row["terminal"] for row in data])
    #             dataset = MDPDataset(observations, actions, rewards, terminals)
    #             os.makedirs(os.path.join(".", "data"), exist_ok=True)
    #             dataset.dump(os.path.join(".", "data", "BreakOut_dataset.h5"))
    #             return dataset
    #         else:
    #             raise NotImplementedError("This data format is not supported at the moment.")
    #     else:
    #         try:
    #             if data_format == "h5":
    #                 return MDPDataset.load(os.path.join(".", "data", "BreakOut_dataset.h5"))
    #             else:
    #                 raise NotImplementedError("This data format is not supported at the moment.")
    #         except Exception:
    #             print("This dataset is not existing, please generate it.")
