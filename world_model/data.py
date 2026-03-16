from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import observation_to_tensor


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    return json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=32)
def _load_npz(path_str: str) -> dict[str, np.ndarray]:
    with np.load(path_str) as data:
        return {name: data[name].copy() for name in data.files}


class FrameDataset(Dataset[torch.Tensor]):
    def __init__(self, dataset_dir: Path, max_frames: int | None = None):
        self.dataset_dir = dataset_dir
        self.manifest = load_manifest(dataset_dir)
        self.index: list[tuple[str, int]] = []

        for episode in self.manifest["episode_files"]:
            episode_path = dataset_dir / episode["path"]
            arrays = _load_npz(str(episode_path))
            count = int(arrays["observations"].shape[0])
            for frame_idx in range(count):
                self.index.append((str(episode_path), frame_idx))
                if max_frames and len(self.index) >= max_frames:
                    return

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> torch.Tensor:
        episode_path, frame_idx = self.index[index]
        arrays = _load_npz(episode_path)
        observation = arrays["observations"][frame_idx]
        return observation_to_tensor(observation)


class LatentSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        dataset_dir: Path,
        *,
        seq_len: int,
        stride: int = 1,
        max_sequences: int | None = None,
    ):
        self.dataset_dir = dataset_dir
        self.manifest = load_manifest(dataset_dir)
        self.seq_len = seq_len
        self.index: list[tuple[str, int]] = []

        for episode in self.manifest["episode_files"]:
            episode_path = dataset_dir / episode["path"]
            arrays = _load_npz(str(episode_path))
            steps = int(arrays["actions"].shape[0])
            if steps < seq_len:
                continue
            max_start = steps - seq_len
            for start in range(0, max_start + 1, stride):
                self.index.append((str(episode_path), start))
                if max_sequences and len(self.index) >= max_sequences:
                    return

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_path, start = self.index[index]
        arrays = _load_npz(episode_path)
        stop = start + self.seq_len

        done_flags = np.logical_or(
            arrays["terminated"][start:stop],
            arrays["truncated"][start:stop],
        )

        return {
            "z": torch.from_numpy(arrays["latents"][start:stop]).float(),
            "next_z": torch.from_numpy(arrays["latents"][start + 1 : stop + 1]).float(),
            "actions": torch.from_numpy(arrays["actions"][start:stop]).long(),
            "rewards": torch.from_numpy(arrays["rewards"][start:stop]).float(),
            "dones": torch.from_numpy(done_flags.astype(np.float32)),
        }
