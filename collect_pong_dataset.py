from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.utils import make_pong_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Atari Pong rollouts and save them in a world-model-friendly "
            "episode dataset."
        )
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to collect. Use 0 to ignore episode count.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="Total number of recorded actions to collect across all environments.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments to collect from concurrently.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=4096,
        help="Maximum number of actions to record per episode before cutting it off.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base seed used to reset each environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "pong_world_model",
        help="Directory that will contain manifest.json and episode files.",
    )
    parser.add_argument(
        "--obs-type",
        choices=("rgb", "grayscale"),
        default="grayscale",
        help="Observation format to save.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=64,
        help="Resize observations to N x N before saving. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--full-action-space",
        action="store_true",
        help="Enable all 18 Atari actions instead of Pong's reduced set.",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Refusing to write into non-empty directory: {output_dir}. "
            "Choose a fresh output path."
        )

    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    return episodes_dir


def save_episode(
    path: Path,
    observations: list[np.ndarray],
    actions: list[int],
    rewards: list[float],
    terminated: list[bool],
    truncated: list[bool],
) -> None:
    np.savez_compressed(
        path,
        observations=np.asarray(observations, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.int16),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
    )


def make_manifest(args: argparse.Namespace, env) -> dict[str, object]:
    return {
        "collector_version": 2,
        "env_id": "ALE/Pong-v5",
        "policy": "random",
        "episodes_requested": args.episodes,
        "total_steps_requested": args.total_steps,
        "num_envs": args.num_envs,
        "max_steps_per_episode": args.max_steps_per_episode,
        "seed": args.seed,
        "obs_type": args.obs_type,
        "resize": args.resize,
        "full_action_space": args.full_action_space,
        "action_space_n": int(env.action_space.n),
        "action_meanings": list(env.unwrapped.get_action_meanings()),
        "episode_files": [],
    }


def new_buffer(
    observation: np.ndarray,
    *,
    episode_seed: int,
) -> dict[str, object]:
    return {
        "observations": [np.array(observation, copy=True)],
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "reward_sum": 0.0,
        "seed": episode_seed,
    }


def empty_buffer() -> dict[str, object]:
    return {
        "observations": [],
        "actions": [],
        "rewards": [],
        "terminated": [],
        "truncated": [],
        "reward_sum": 0.0,
        "seed": -1,
    }


def should_continue(
    args: argparse.Namespace,
    *,
    steps_collected: int,
    episodes_collected: int,
) -> bool:
    episodes_ok = args.episodes <= 0 or episodes_collected < args.episodes
    steps_ok = args.total_steps <= 0 or steps_collected < args.total_steps
    return episodes_ok and steps_ok


def finalize_episode(
    *,
    episodes_dir: Path,
    output_dir: Path,
    manifest: dict[str, object],
    episode_index: int,
    buffer: dict[str, object],
    end_reason: str,
) -> int:
    actions = buffer["actions"]
    if not actions:
        return 0

    episode_path = episodes_dir / f"episode_{episode_index:06d}.npz"
    save_episode(
        episode_path,
        buffer["observations"],
        actions,
        buffer["rewards"],
        buffer["terminated"],
        buffer["truncated"],
    )

    steps = len(actions)
    manifest["episode_files"].append(
        {
            "episode_index": episode_index,
            "path": episode_path.relative_to(output_dir).as_posix(),
            "seed": int(buffer["seed"]),
            "steps": steps,
            "reward_sum": float(buffer["reward_sum"]),
            "end_reason": end_reason,
        }
    )

    print(
        f"[episode {episode_index + 1}] saved {episode_path.name} | "
        f"steps={steps} | reward={float(buffer['reward_sum']):.2f} | end={end_reason}"
    )
    return steps


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 and args.total_steps <= 0:
        raise SystemExit("Set --episodes or --total-steps to a positive value.")

    episodes_dir = prepare_output_dir(args.output_dir)
    envs = [
        make_pong_env(
            obs_type=args.obs_type,
            resize=args.resize,
            full_action_space=args.full_action_space,
        )
        for _ in range(args.num_envs)
    ]
    manifest = make_manifest(args, envs[0])

    observations: list[np.ndarray] = []
    buffers: list[dict[str, object]] = []
    next_seed = args.seed

    for env in envs:
        observation, _ = env.reset(seed=next_seed)
        observations.append(observation)
        buffers.append(new_buffer(observation, episode_seed=next_seed))
        next_seed += 1

    if observations:
        manifest["observation_shape"] = list(observations[0].shape)
        manifest["observation_dtype"] = str(observations[0].dtype)

    steps_collected = 0
    episode_index = 0

    try:
        while should_continue(
            args,
            steps_collected=steps_collected,
            episodes_collected=episode_index,
        ):
            for env_idx, env in enumerate(envs):
                if not should_continue(
                    args,
                    steps_collected=steps_collected,
                    episodes_collected=episode_index,
                ):
                    break

                action = int(env.action_space.sample())
                observation, reward, terminated, truncated, _ = env.step(action)
                buffer = buffers[env_idx]

                buffer["observations"].append(np.array(observation, copy=True))
                buffer["actions"].append(action)
                buffer["rewards"].append(float(reward))
                buffer["terminated"].append(bool(terminated))
                buffer["truncated"].append(bool(truncated))
                buffer["reward_sum"] = float(buffer["reward_sum"]) + float(reward)
                observations[env_idx] = observation
                steps_collected += 1

                forced_cutoff = len(buffer["actions"]) >= args.max_steps_per_episode
                if terminated or truncated or forced_cutoff:
                    if terminated:
                        end_reason = "terminated"
                    elif truncated:
                        end_reason = "truncated"
                    else:
                        end_reason = "collector_cutoff"

                    finalize_episode(
                        episodes_dir=episodes_dir,
                        output_dir=args.output_dir,
                        manifest=manifest,
                        episode_index=episode_index,
                        buffer=buffer,
                        end_reason=end_reason,
                    )
                    episode_index += 1

                    if should_continue(
                        args,
                        steps_collected=steps_collected,
                        episodes_collected=episode_index,
                    ):
                        observation, _ = env.reset(seed=next_seed)
                        observations[env_idx] = observation
                        buffers[env_idx] = new_buffer(observation, episode_seed=next_seed)
                        next_seed += 1
                    else:
                        buffers[env_idx] = empty_buffer()

        for buffer in buffers:
            if should_continue(
                args,
                steps_collected=steps_collected,
                episodes_collected=episode_index,
            ):
                break
            episode_steps = finalize_episode(
                episodes_dir=episodes_dir,
                output_dir=args.output_dir,
                manifest=manifest,
                episode_index=episode_index,
                buffer=buffer,
                end_reason="final_flush",
            )
            if episode_steps > 0:
                episode_index += 1
    finally:
        for env in envs:
            env.close()

    episode_lengths = [episode["steps"] for episode in manifest["episode_files"]]
    total_reward = sum(float(episode["reward_sum"]) for episode in manifest["episode_files"])
    manifest["summary"] = {
        "episodes_collected": len(manifest["episode_files"]),
        "total_steps": int(sum(episode_lengths)),
        "total_reward": float(total_reward),
        "mean_episode_length": (
            float(np.mean(episode_lengths)) if episode_lengths else 0.0
        ),
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(
        f"Dataset summary: episodes={len(manifest['episode_files'])}, "
        f"steps={int(sum(episode_lengths))}, total_reward={float(total_reward):.2f}"
    )


if __name__ == "__main__":
    main()
