from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from world_model.data import LatentSequenceDataset
from world_model.models import MDNRNN, mdn_loss
from world_model.utils import (
    maybe_autocast,
    pick_device,
    save_checkpoint,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MDN-RNN on latent Pong rollouts.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-mixtures", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action-embed-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--done-weight", type=float, default=0.2)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def sequence_loss(
    outputs: dict[str, torch.Tensor],
    next_z: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    done_weight: float,
    reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nll = mdn_loss(
        outputs["mixture_logits"],
        outputs["mixture_mu"],
        outputs["mixture_logstd"],
        next_z,
    )
    done_loss = F.binary_cross_entropy_with_logits(outputs["done_logits"], dones)
    reward_loss = F.smooth_l1_loss(outputs["reward"], rewards)
    total = nll + done_weight * done_loss + reward_weight * reward_loss
    return total, nll, done_loss, reward_loss


def evaluate(
    model: MDNRNN,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_nll = 0.0
    total_done = 0.0
    total_reward = 0.0
    count = 0
    use_amp = device.type == "cuda" and not args.no_amp

    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device, non_blocking=True)
            next_z = batch["next_z"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            rewards = batch["rewards"].to(device, non_blocking=True)
            dones = batch["dones"].to(device, non_blocking=True)
            with maybe_autocast(device, use_amp):
                outputs, _ = model(z, actions)
                loss, nll, done_loss, reward_loss = sequence_loss(
                    outputs,
                    next_z,
                    rewards,
                    dones,
                    done_weight=args.done_weight,
                    reward_weight=args.reward_weight,
                )

            batch_size = z.size(0)
            total_loss += float(loss.item()) * batch_size
            total_nll += float(nll.item()) * batch_size
            total_done += float(done_loss.item()) * batch_size
            total_reward += float(reward_loss.item()) * batch_size
            count += batch_size

    denom = max(count, 1)
    return (
        total_loss / denom,
        total_nll / denom,
        total_done / denom,
        total_reward / denom,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = LatentSequenceDataset(
        args.dataset_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        max_sequences=args.max_sequences if args.max_sequences > 0 else None,
    )
    if len(dataset) < 2:
        raise SystemExit("Need at least 2 latent sequences to train the MDN-RNN.")

    latent_dim = int(manifest["latent_dim"])
    action_space_n = int(manifest["action_space_n"])

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = MDNRNN(
        latent_dim=latent_dim,
        action_space_n=action_space_n,
        hidden_size=args.hidden_size,
        num_mixtures=args.num_mixtures,
        action_embed_dim=args.action_embed_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    train_model = model
    if args.compile and hasattr(torch, "compile"):
        train_model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_done = 0.0
        total_reward = 0.0
        count = 0

        for batch in train_loader:
            z = batch["z"].to(device, non_blocking=True)
            next_z = batch["next_z"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            rewards = batch["rewards"].to(device, non_blocking=True)
            dones = batch["dones"].to(device, non_blocking=True)

            with maybe_autocast(device, use_amp):
                outputs, _ = train_model(z, actions)
                loss, nll, done_loss, reward_loss = sequence_loss(
                    outputs,
                    next_z,
                    rewards,
                    dones,
                    done_weight=args.done_weight,
                    reward_weight=args.reward_weight,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = z.size(0)
            total_loss += float(loss.item()) * batch_size
            total_nll += float(nll.item()) * batch_size
            total_done += float(done_loss.item()) * batch_size
            total_reward += float(reward_loss.item()) * batch_size
            count += batch_size

        train_loss = total_loss / max(count, 1)
        val_loss, val_nll, val_done, val_reward = evaluate(
            train_model,
            val_loader,
            device,
            args,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_nll": total_nll / max(count, 1),
                "train_done_loss": total_done / max(count, 1),
                "train_reward_loss": total_reward / max(count, 1),
                "val_loss": val_loss,
                "val_nll": val_nll,
                "val_done_loss": val_done,
                "val_reward_loss": val_reward,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss <= best_val:
            best_val = val_loss
            save_checkpoint(
                args.output_dir / "best.pt",
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "latent_dim": latent_dim,
                        "action_space_n": action_space_n,
                        "hidden_size": args.hidden_size,
                        "num_mixtures": args.num_mixtures,
                        "action_embed_dim": args.action_embed_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "obs_type": manifest["obs_type"],
                        "resize": manifest["resize"],
                        "full_action_space": manifest["full_action_space"],
                    },
                },
            )

    save_checkpoint(
        args.output_dir / "last.pt",
        {
            "model_state": model.state_dict(),
            "config": {
                "latent_dim": latent_dim,
                "action_space_n": action_space_n,
                "hidden_size": args.hidden_size,
                "num_mixtures": args.num_mixtures,
                "action_embed_dim": args.action_embed_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "obs_type": manifest["obs_type"],
                "resize": manifest["resize"],
                "full_action_space": manifest["full_action_space"],
            },
        },
    )
    write_json(args.output_dir / "history.json", {"epochs": history, "best_val": best_val})


if __name__ == "__main__":
    main()
