from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from world_model.models import build_vae_from_config
from world_model.utils import (
    load_checkpoint,
    maybe_autocast,
    observation_batch_to_tensor,
    pick_device,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode Pong episodes into VAE latents.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    use_amp = device.type == "cuda"

    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    model = build_vae_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))

    latent_manifest = {
        "source_dataset_dir": str(args.dataset_dir),
        "vae_checkpoint": str(args.checkpoint),
        "latent_dim": config["latent_dim"],
        "action_space_n": manifest["action_space_n"],
        "action_meanings": manifest["action_meanings"],
        "obs_type": manifest["obs_type"],
        "resize": manifest["resize"],
        "full_action_space": manifest["full_action_space"],
        "episode_files": [],
    }

    total_latents = 0

    for episode in manifest["episode_files"]:
        source_path = args.dataset_dir / episode["path"]
        with np.load(source_path) as data:
            observations = data["observations"]
            actions = data["actions"]
            rewards = data["rewards"]
            terminated = data["terminated"]
            truncated = data["truncated"]

        latent_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, observations.shape[0], args.batch_size):
                batch = observations[start : start + args.batch_size]
                batch_tensor = observation_batch_to_tensor(batch).to(device)
                with maybe_autocast(device, use_amp):
                    latents = model.encode_mean(batch_tensor)
                latent_batches.append(latents.cpu().numpy())

        latent_array = np.concatenate(latent_batches, axis=0).astype(np.float32)
        total_latents += int(latent_array.shape[0])

        relative_path = Path(episode["path"])
        target_path = args.output_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target_path,
            latents=latent_array,
            actions=actions.astype(np.int16),
            rewards=rewards.astype(np.float32),
            terminated=terminated.astype(np.bool_),
            truncated=truncated.astype(np.bool_),
        )

        latent_manifest["episode_files"].append(
            {
                "episode_index": episode["episode_index"],
                "path": relative_path.as_posix(),
                "steps": episode["steps"],
                "reward_sum": episode["reward_sum"],
                "end_reason": episode["end_reason"],
            }
        )

        print(
            f"encoded {relative_path.name} | latents={latent_array.shape[0]} "
            f"| steps={actions.shape[0]}"
        )

    latent_manifest["summary"] = {
        "episodes_collected": len(latent_manifest["episode_files"]),
        "total_latents": total_latents,
    }
    write_json(args.output_dir / "manifest.json", latent_manifest)


if __name__ == "__main__":
    main()
