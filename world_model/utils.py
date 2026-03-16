from __future__ import annotations

from contextlib import nullcontext
import json
import random
from pathlib import Path
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
import torch


ENV_ID = "ALE/Pong-v5"


class NearestResizeObservation(gym.ObservationWrapper[Any, Any, Any]):
    def __init__(self, env: gym.Env[Any, Any], size: int):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("NearestResizeObservation requires a Box observation space.")
        if len(env.observation_space.shape) not in (2, 3):
            raise ValueError("Expected grayscale or RGB image observations.")

        source_shape = env.observation_space.shape
        resized_shape = (
            (size, size)
            if len(source_shape) == 2
            else (size, size, source_shape[2])
        )

        self._row_idx = np.linspace(0, source_shape[0] - 1, size).astype(np.int32)
        self._col_idx = np.linspace(0, source_shape[1] - 1, size).astype(np.int32)
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=resized_shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        resized = observation[self._row_idx][:, self._col_idx]
        return np.asarray(resized, dtype=self.observation_space.dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or raw == "":
        return list(default)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def maybe_autocast(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def zero_hidden_state(
    hidden: tuple[torch.Tensor, torch.Tensor],
    done_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if done_mask.ndim != 1:
        raise ValueError("done_mask must be a 1D tensor of shape [batch].")
    if not torch.any(done_mask):
        return hidden

    next_hidden = []
    for tensor in hidden:
        updated = tensor.clone()
        updated[:, done_mask] = 0.0
        next_hidden.append(updated)
    return next_hidden[0], next_hidden[1]


def make_pong_env(
    *,
    obs_type: str = "grayscale",
    resize: int = 64,
    full_action_space: bool = False,
) -> gym.Env[Any, Any]:
    gym.register_envs(ale_py)

    env = gym.make(
        ENV_ID,
        obs_type=obs_type,
        full_action_space=full_action_space,
    )

    if resize > 0:
        env = NearestResizeObservation(env, resize)

    return env


def observation_to_tensor(observation: np.ndarray) -> torch.Tensor:
    if observation.ndim == 2:
        tensor = torch.from_numpy(observation).unsqueeze(0)
    elif observation.ndim == 3:
        tensor = torch.from_numpy(np.transpose(observation, (2, 0, 1)))
    else:
        raise ValueError(f"Unsupported observation shape: {observation.shape}")
    return tensor.float() / 255.0


def observation_batch_to_tensor(observations: np.ndarray) -> torch.Tensor:
    if observations.ndim == 3:
        tensor = torch.from_numpy(observations).unsqueeze(1)
    elif observations.ndim == 4:
        tensor = torch.from_numpy(np.transpose(observations, (0, 3, 1, 2)))
    else:
        raise ValueError(f"Unsupported observation batch shape: {observations.shape}")
    return tensor.float() / 255.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)
