from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILES = {
    "smoke": {
        "collect_total_steps": 8_192,
        "collect_num_envs": 2,
        "collect_max_steps_per_episode": 256,
        "obs_type": "grayscale",
        "resize": 64,
        "vae_epochs": 3,
        "vae_latent_dim": 64,
        "vae_hidden_dims": "64,128,256,512",
        "vae_residual_blocks": 1,
        "vae_batch_size": 128,
        "vae_num_workers": 0,
        "rnn_epochs": 3,
        "rnn_hidden_size": 512,
        "rnn_num_layers": 1,
        "rnn_num_mixtures": 5,
        "rnn_batch_size": 128,
        "rnn_num_workers": 0,
        "controller_num_envs": 4,
        "controller_rollout_steps": 128,
        "controller_total_timesteps": 16_384,
        "controller_hidden_dims": "256,256",
    },
    "h200": {
        "collect_total_steps": 1_500_000,
        "collect_num_envs": 16,
        "collect_max_steps_per_episode": 4096,
        "obs_type": "grayscale",
        "resize": 64,
        "vae_epochs": 40,
        "vae_latent_dim": 128,
        "vae_hidden_dims": "96,192,384,768",
        "vae_residual_blocks": 2,
        "vae_batch_size": 1024,
        "vae_num_workers": 8,
        "rnn_epochs": 30,
        "rnn_hidden_size": 1024,
        "rnn_num_layers": 2,
        "rnn_num_mixtures": 8,
        "rnn_batch_size": 512,
        "rnn_num_workers": 8,
        "controller_num_envs": 16,
        "controller_rollout_steps": 256,
        "controller_total_timesteps": 5_000_000,
        "controller_hidden_dims": "512,512",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Pong world-model pipeline end to end."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.preset]
    python = sys.executable

    dataset_dir = args.run_dir / "dataset"
    vae_dir = args.run_dir / "vae"
    latent_dir = args.run_dir / "latents"
    rnn_dir = args.run_dir / "mdn_rnn"
    controller_dir = args.run_dir / "controller"

    run(
        [
            python,
            "collect_pong_dataset.py",
            "--total-steps",
            str(profile["collect_total_steps"]),
            "--num-envs",
            str(profile["collect_num_envs"]),
            "--max-steps-per-episode",
            str(profile["collect_max_steps_per_episode"]),
            "--output-dir",
            str(dataset_dir),
            "--obs-type",
            str(profile["obs_type"]),
            "--resize",
            str(profile["resize"]),
            "--seed",
            str(args.seed),
        ]
    )
    run(
        [
            python,
            "train_vae.py",
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(vae_dir),
            "--latent-dim",
            str(profile["vae_latent_dim"]),
            "--hidden-dims",
            str(profile["vae_hidden_dims"]),
            "--residual-blocks",
            str(profile["vae_residual_blocks"]),
            "--batch-size",
            str(profile["vae_batch_size"]),
            "--num-workers",
            str(profile["vae_num_workers"]),
            "--epochs",
            str(profile["vae_epochs"]),
            "--seed",
            str(args.seed),
        ]
    )
    run(
        [
            python,
            "encode_latents.py",
            "--dataset-dir",
            str(dataset_dir),
            "--checkpoint",
            str(vae_dir / "best.pt"),
            "--output-dir",
            str(latent_dir),
        ]
    )
    run(
        [
            python,
            "train_mdn_rnn.py",
            "--dataset-dir",
            str(latent_dir),
            "--output-dir",
            str(rnn_dir),
            "--epochs",
            str(profile["rnn_epochs"]),
            "--batch-size",
            str(profile["rnn_batch_size"]),
            "--num-workers",
            str(profile["rnn_num_workers"]),
            "--hidden-size",
            str(profile["rnn_hidden_size"]),
            "--num-layers",
            str(profile["rnn_num_layers"]),
            "--num-mixtures",
            str(profile["rnn_num_mixtures"]),
            "--seed",
            str(args.seed),
        ]
    )
    run(
        [
            python,
            "train_controller.py",
            "--vae-checkpoint",
            str(vae_dir / "best.pt"),
            "--rnn-checkpoint",
            str(rnn_dir / "best.pt"),
            "--output-dir",
            str(controller_dir),
            "--num-envs",
            str(profile["controller_num_envs"]),
            "--rollout-steps",
            str(profile["controller_rollout_steps"]),
            "--total-timesteps",
            str(profile["controller_total_timesteps"]),
            "--hidden-dims",
            str(profile["controller_hidden_dims"]),
            "--seed",
            str(args.seed),
        ]
    )


if __name__ == "__main__":
    main()
