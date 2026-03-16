from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from world_model.models import (
    ActorCriticController,
    build_rnn_from_config,
    build_vae_from_config,
)
from world_model.utils import (
    load_checkpoint,
    make_pong_env,
    maybe_autocast,
    observation_batch_to_tensor,
    parse_int_list,
    pick_device,
    save_checkpoint,
    seed_everything,
    write_json,
    zero_hidden_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO controller on Pong using frozen VAE + MDN-RNN features."
    )
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--rnn-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dims", type=str, default="512,512")
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def evaluate_policy(
    controller: ActorCriticController,
    vae,
    rnn,
    *,
    device: torch.device,
    use_amp: bool,
    episodes: int,
    seed: int,
    obs_type: str,
    resize: int,
    full_action_space: bool,
) -> float:
    returns: list[float] = []
    env = make_pong_env(
        obs_type=obs_type,
        resize=resize,
        full_action_space=full_action_space,
    )

    try:
        for episode_idx in range(episodes):
            observation, _ = env.reset(seed=seed + episode_idx)
            hidden = rnn.init_hidden(batch_size=1, device=device)
            total_reward = 0.0

            while True:
                batch = np.expand_dims(observation, axis=0)
                with torch.no_grad():
                    with maybe_autocast(device, use_amp):
                        z = vae.encode_mean(
                            observation_batch_to_tensor(batch).to(device)
                        )
                    features = torch.cat([z, hidden[0][-1]], dim=-1)
                    logits, _ = controller(features)
                    action = torch.argmax(logits, dim=-1)
                    _, hidden = rnn.forward_step(z, action, hidden)

                observation, reward, terminated, truncated, _ = env.step(int(action.item()))
                total_reward += float(reward)
                if terminated or truncated:
                    returns.append(total_reward)
                    break
    finally:
        env.close()

    return float(np.mean(returns)) if returns else 0.0


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    vae_checkpoint = load_checkpoint(args.vae_checkpoint, device)
    rnn_checkpoint = load_checkpoint(args.rnn_checkpoint, device)

    vae = build_vae_from_config(vae_checkpoint["config"]).to(device)
    vae.load_state_dict(vae_checkpoint["model_state"])
    vae.eval()

    rnn = build_rnn_from_config(rnn_checkpoint["config"]).to(device)
    rnn.load_state_dict(rnn_checkpoint["model_state"])
    rnn.eval()

    hidden_dims = parse_int_list(args.hidden_dims, [512, 512])
    controller = ActorCriticController(
        input_dim=int(vae_checkpoint["config"]["latent_dim"])
        + int(rnn_checkpoint["config"]["hidden_size"]),
        action_space_n=int(rnn_checkpoint["config"]["action_space_n"]),
        hidden_dims=hidden_dims,
    ).to(device)
    controller_model = controller
    if args.compile and hasattr(torch, "compile"):
        controller_model = torch.compile(controller)

    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate)

    envs = [
        make_pong_env(
            obs_type=rnn_checkpoint["config"]["obs_type"],
            resize=int(rnn_checkpoint["config"]["resize"]),
            full_action_space=bool(rnn_checkpoint["config"]["full_action_space"]),
        )
        for _ in range(args.num_envs)
    ]

    try:
        observations = []
        next_seed = args.seed
        for env in envs:
            observation, _ = env.reset(seed=next_seed)
            observations.append(observation)
            next_seed += 1

        hidden = rnn.init_hidden(batch_size=args.num_envs, device=device)
        episode_returns = np.zeros(args.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []

        steps_per_update = args.num_envs * args.rollout_steps
        num_updates = max(1, args.total_timesteps // steps_per_update)
        history: list[dict[str, float]] = []
        best_eval_return = float("-inf")

        feature_dim = int(vae_checkpoint["config"]["latent_dim"]) + int(
            rnn_checkpoint["config"]["hidden_size"]
        )

        for update in range(1, num_updates + 1):
            controller_model.train()
            features_buf = torch.zeros(
                (args.rollout_steps, args.num_envs, feature_dim),
                device=device,
            )
            actions_buf = torch.zeros(
                (args.rollout_steps, args.num_envs),
                dtype=torch.long,
                device=device,
            )
            logprobs_buf = torch.zeros(
                (args.rollout_steps, args.num_envs),
                device=device,
            )
            rewards_buf = torch.zeros(
                (args.rollout_steps, args.num_envs),
                device=device,
            )
            dones_buf = torch.zeros(
                (args.rollout_steps, args.num_envs),
                device=device,
            )
            values_buf = torch.zeros(
                (args.rollout_steps, args.num_envs),
                device=device,
            )

            for step in range(args.rollout_steps):
                obs_batch = np.stack(observations, axis=0)
                with torch.no_grad():
                    with maybe_autocast(device, use_amp):
                        z = vae.encode_mean(
                            observation_batch_to_tensor(obs_batch).to(device)
                        )
                    features = torch.cat([z, hidden[0][-1]], dim=-1)
                    logits, values = controller_model(features)
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    logprobs = dist.log_prob(actions)

                features_buf[step] = features
                actions_buf[step] = actions
                logprobs_buf[step] = logprobs
                values_buf[step] = values

                next_observations: list[np.ndarray] = []
                done_mask_list: list[bool] = []
                reward_list: list[float] = []

                for env_idx, env in enumerate(envs):
                    next_observation, reward, terminated, truncated, _ = env.step(
                        int(actions[env_idx].item())
                    )
                    done = bool(terminated or truncated)
                    episode_returns[env_idx] += float(reward)
                    episode_lengths[env_idx] += 1

                    if done:
                        completed_returns.append(float(episode_returns[env_idx]))
                        completed_lengths.append(int(episode_lengths[env_idx]))
                        next_observation, _ = env.reset(seed=next_seed)
                        next_seed += 1
                        episode_returns[env_idx] = 0.0
                        episode_lengths[env_idx] = 0

                    next_observations.append(next_observation)
                    reward_list.append(float(reward))
                    done_mask_list.append(done)

                rewards_buf[step] = torch.tensor(reward_list, device=device)
                done_mask = torch.tensor(done_mask_list, device=device, dtype=torch.bool)
                dones_buf[step] = done_mask.float()

                with torch.no_grad():
                    _, hidden = rnn.forward_step(z, actions, hidden)
                    hidden = zero_hidden_state(hidden, done_mask)

                observations = next_observations

            with torch.no_grad():
                obs_batch = np.stack(observations, axis=0)
                with maybe_autocast(device, use_amp):
                    z = vae.encode_mean(
                        observation_batch_to_tensor(obs_batch).to(device)
                    )
                next_features = torch.cat([z, hidden[0][-1]], dim=-1)
                _, next_values = controller_model(next_features)

            advantages = torch.zeros_like(rewards_buf)
            last_advantage = torch.zeros(args.num_envs, device=device)
            for step in reversed(range(args.rollout_steps)):
                if step == args.rollout_steps - 1:
                    next_value = next_values
                else:
                    next_value = values_buf[step + 1]

                next_non_terminal = 1.0 - dones_buf[step]
                delta = (
                    rewards_buf[step]
                    + args.gamma * next_value * next_non_terminal
                    - values_buf[step]
                )
                last_advantage = (
                    delta
                    + args.gamma * args.gae_lambda * next_non_terminal * last_advantage
                )
                advantages[step] = last_advantage

            returns = advantages + values_buf

            flat_features = features_buf.reshape(-1, feature_dim)
            flat_actions = actions_buf.reshape(-1)
            flat_logprobs = logprobs_buf.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)

            flat_advantages = (flat_advantages - flat_advantages.mean()) / (
                flat_advantages.std(unbiased=False) + 1e-6
            )

            batch_size = flat_features.size(0)
            permutation = torch.arange(batch_size, device=device)

            for _ in range(args.ppo_epochs):
                permutation = permutation[torch.randperm(batch_size, device=device)]
                for start in range(0, batch_size, args.minibatch_size):
                    indices = permutation[start : start + args.minibatch_size]
                    mb_features = flat_features[indices]
                    mb_actions = flat_actions[indices]
                    mb_old_logprobs = flat_logprobs[indices]
                    mb_advantages = flat_advantages[indices]
                    mb_returns = flat_returns[indices]

                    logits, values = controller_model(mb_features)
                    dist = Categorical(logits=logits)
                    new_logprobs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    ratio = (new_logprobs - mb_old_logprobs).exp()
                    policy_loss_1 = -mb_advantages * ratio
                    policy_loss_2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - args.clip_coef,
                        1.0 + args.clip_coef,
                    )
                    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                    value_loss = 0.5 * F.mse_loss(values, mb_returns)
                    loss = (
                        policy_loss
                        + args.vf_coef * value_loss
                        - args.entropy_coef * entropy
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        controller.parameters(),
                        max_norm=args.max_grad_norm,
                    )
                    optimizer.step()

            train_mean_return = (
                float(np.mean(completed_returns[-100:])) if completed_returns else 0.0
            )
            train_mean_length = (
                float(np.mean(completed_lengths[-100:])) if completed_lengths else 0.0
            )

            history_entry = {
                "update": update,
                "timesteps": int(update * steps_per_update),
                "train_mean_return": train_mean_return,
                "train_mean_length": train_mean_length,
            }

            if update % args.eval_interval == 0 or update == num_updates:
                eval_return = evaluate_policy(
                    controller_model,
                    vae,
                    rnn,
                    device=device,
                    use_amp=use_amp,
                    episodes=args.eval_episodes,
                    seed=args.seed + 100_000 + update * args.eval_episodes,
                    obs_type=rnn_checkpoint["config"]["obs_type"],
                    resize=int(rnn_checkpoint["config"]["resize"]),
                    full_action_space=bool(
                        rnn_checkpoint["config"]["full_action_space"]
                    ),
                )
                history_entry["eval_mean_return"] = eval_return
                if eval_return >= best_eval_return:
                    best_eval_return = eval_return
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        {
                            "model_state": controller.state_dict(),
                            "config": {
                                "input_dim": feature_dim,
                                "action_space_n": int(
                                    rnn_checkpoint["config"]["action_space_n"]
                                ),
                                "hidden_dims": hidden_dims,
                            },
                        },
                    )
                print(
                    f"update={update}/{num_updates} "
                    f"train_mean_return={train_mean_return:.2f} "
                    f"eval_mean_return={eval_return:.2f}"
                )
                controller_model.train()
            else:
                print(
                    f"update={update}/{num_updates} "
                    f"train_mean_return={train_mean_return:.2f}"
                )

            history.append(history_entry)

    finally:
        for env in envs:
            env.close()

    save_checkpoint(
        args.output_dir / "last.pt",
        {
            "model_state": controller.state_dict(),
            "config": {
                "input_dim": feature_dim,
                "action_space_n": int(rnn_checkpoint["config"]["action_space_n"]),
                "hidden_dims": hidden_dims,
            },
        },
    )
    write_json(
        args.output_dir / "history.json",
        {
            "updates": history,
            "best_eval_return": best_eval_return,
        },
    )


if __name__ == "__main__":
    main()
