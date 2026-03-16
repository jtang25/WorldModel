from __future__ import annotations

import argparse

import ale_py
import gymnasium as gym


ENV_ID = "ALE/Pong-v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Atari Pong through Gymnasium + ALE and run random actions."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Maximum number of random actions to execute.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed used for env.reset and action_space.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array"),
        default="rgb_array",
        help="Use human to open a window, or rgb_array for a headless smoke test.",
    )
    parser.add_argument(
        "--full-action-space",
        action="store_true",
        help="Enable all 18 Atari actions instead of Pong's reduced set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gym.register_envs(ale_py)

    env = gym.make(
        ENV_ID,
        render_mode=args.render_mode,
        full_action_space=args.full_action_space,
    )
    env.action_space.seed(args.seed)

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    steps_run = 0

    print(f"Loaded {ENV_ID}")
    print(f"Action space: {env.action_space}")
    print(f"Observation shape: {getattr(obs, 'shape', type(obs))}")
    print(f"Render mode: {args.render_mode}")

    try:
        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps_run = step + 1

            if terminated or truncated:
                print(
                    f"Episode ended after {steps_run} steps "
                    f"(terminated={terminated}, truncated={truncated}). Resetting."
                )
                obs, info = env.reset()
    finally:
        env.close()

    print(f"Completed {steps_run} steps with total reward {total_reward:.2f}")


if __name__ == "__main__":
    main()

