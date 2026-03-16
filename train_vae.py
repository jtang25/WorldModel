from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from world_model.data import FrameDataset
from world_model.models import ConvVAE, vae_loss
from world_model.utils import (
    maybe_autocast,
    parse_int_list,
    pick_device,
    save_checkpoint,
    seed_everything,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on Pong rollout frames.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", type=str, default="96,192,384,768")
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--recon-loss", choices=("bce", "mse", "smooth_l1"), default="bce")
    parser.add_argument("--foreground-weight", type=float, default=8.0)
    parser.add_argument("--foreground-threshold", type=float, default=0.05)
    parser.add_argument("--free-nats", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def current_beta(args: argparse.Namespace, epoch: int) -> float:
    if args.kl_warmup_epochs <= 0:
        return args.beta
    return args.beta * min(epoch / args.kl_warmup_epochs, 1.0)


def evaluate(
    model: ConvVAE,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    beta: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    count = 0
    use_amp = device.type == "cuda" and not args.no_amp

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with maybe_autocast(device, use_amp):
                reconstruction, mu, logvar, _, logits = model(batch)
                loss, recon, kl = vae_loss(
                    reconstruction,
                    logits,
                    batch,
                    mu,
                    logvar,
                    beta=beta,
                    recon_loss_type=args.recon_loss,
                    foreground_weight=args.foreground_weight,
                    foreground_threshold=args.foreground_threshold,
                    free_nats=args.free_nats,
                )
            batch_size = batch.size(0)
            total_loss += float(loss.item()) * batch_size
            total_recon += float(recon.item()) * batch_size
            total_kl += float(kl.item()) * batch_size
            count += batch_size

    denom = max(count, 1)
    return total_loss / denom, total_recon / denom, total_kl / denom


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    hidden_dims = parse_int_list(args.hidden_dims, [96, 192, 384, 768])

    dataset = FrameDataset(
        args.dataset_dir,
        max_frames=args.max_frames if args.max_frames > 0 else None,
    )
    if len(dataset) < 2:
        raise SystemExit("Need at least 2 frames to train the VAE.")

    sample = dataset[0]
    input_channels = int(sample.shape[0])
    image_height = int(sample.shape[1])
    image_width = int(sample.shape[2])
    if image_height != image_width:
        raise SystemExit("The current VAE expects square observations.")

    dataset_manifest = json.loads(
        (args.dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )

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

    model = ConvVAE(
        input_channels=input_channels,
        image_size=image_height,
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
        residual_blocks=args.residual_blocks,
    ).to(device)
    train_model = model
    if args.compile and hasattr(torch, "compile"):
        train_model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    use_amp = device.type == "cuda" and not args.no_amp

    for epoch in range(1, args.epochs + 1):
        beta = current_beta(args, epoch)
        train_model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        count = 0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)

            with maybe_autocast(device, use_amp):
                reconstruction, mu, logvar, _, logits = train_model(batch)
                loss, recon, kl = vae_loss(
                    reconstruction,
                    logits,
                    batch,
                    mu,
                    logvar,
                    beta=beta,
                    recon_loss_type=args.recon_loss,
                    foreground_weight=args.foreground_weight,
                    foreground_threshold=args.foreground_threshold,
                    free_nats=args.free_nats,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = batch.size(0)
            total_loss += float(loss.item()) * batch_size
            total_recon += float(recon.item()) * batch_size
            total_kl += float(kl.item()) * batch_size
            count += batch_size

        train_loss = total_loss / max(count, 1)
        train_recon = total_recon / max(count, 1)
        train_kl = total_kl / max(count, 1)
        val_loss, val_recon, val_kl = evaluate(train_model, val_loader, device, args, beta)

        history.append(
            {
                "epoch": epoch,
                "beta": beta,
                "train_loss": train_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_loss": val_loss,
                "val_recon": val_recon,
                "val_kl": val_kl,
            }
        )

        print(
            f"epoch={epoch} beta={beta:.6f} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss <= best_val:
            best_val = val_loss
            save_checkpoint(
                args.output_dir / "best.pt",
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "latent_dim": args.latent_dim,
                        "input_channels": input_channels,
                        "dataset_dir": str(args.dataset_dir),
                        "obs_type": dataset_manifest["obs_type"],
                        "resize": dataset_manifest["resize"],
                        "full_action_space": dataset_manifest["full_action_space"],
                        "image_size": [image_height, image_width],
                        "hidden_dims": hidden_dims,
                        "residual_blocks": args.residual_blocks,
                        "recon_loss": args.recon_loss,
                    },
                },
            )

    save_checkpoint(
        args.output_dir / "last.pt",
        {
            "model_state": model.state_dict(),
            "config": {
                "latent_dim": args.latent_dim,
                "input_channels": input_channels,
                "dataset_dir": str(args.dataset_dir),
                "obs_type": dataset_manifest["obs_type"],
                "resize": dataset_manifest["resize"],
                "full_action_space": dataset_manifest["full_action_space"],
                "image_size": [image_height, image_width],
                "hidden_dims": hidden_dims,
                "residual_blocks": args.residual_blocks,
                "recon_loss": args.recon_loss,
            },
        },
    )
    write_json(args.output_dir / "history.json", {"epochs": history, "best_val": best_val})


if __name__ == "__main__":
    main()
