from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from world_model.models import build_rnn_from_config, build_vae_from_config
from world_model.utils import (
    load_checkpoint,
    maybe_autocast,
    observation_batch_to_tensor,
    pick_device,
)


ROW_ORDER = [
    "current_frame",
    "vae_reconstruction",
    "true_next_frame",
    "predicted_next_frame",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a side-by-side visualization of what the trained world model "
            "imagines for the next Pong frame."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--rnn-checkpoint", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_models(
    vae_checkpoint_path: Path,
    rnn_checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    vae_checkpoint = load_checkpoint(vae_checkpoint_path, device)
    rnn_checkpoint = load_checkpoint(rnn_checkpoint_path, device)

    vae = build_vae_from_config(vae_checkpoint["config"]).to(device)
    vae.load_state_dict(vae_checkpoint["model_state"])
    vae.eval()

    rnn = build_rnn_from_config(rnn_checkpoint["config"]).to(device)
    rnn.load_state_dict(rnn_checkpoint["model_state"])
    rnn.eval()

    return vae, rnn


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().squeeze(0)
    if image.ndim == 2:
        array = image.numpy()
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.repeat(array[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[0] == 1:
        array = image[0].numpy()
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.repeat(array[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[0] == 3:
        array = np.transpose(image.numpy(), (1, 2, 0))
        return np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    raise ValueError(f"Unsupported tensor shape: {tuple(image.shape)}")


def make_contact_sheet(
    rows: list[list[np.ndarray]],
    *,
    padding: int = 2,
    background: int = 245,
) -> np.ndarray:
    tile_height, tile_width, _ = rows[0][0].shape
    num_rows = len(rows)
    num_cols = len(rows[0])
    height = num_rows * tile_height + (num_rows + 1) * padding
    width = num_cols * tile_width + (num_cols + 1) * padding
    canvas = np.full((height, width, 3), background, dtype=np.uint8)

    for row_idx, row in enumerate(rows):
        for col_idx, tile in enumerate(row):
            top = padding + row_idx * (tile_height + padding)
            left = padding + col_idx * (tile_width + padding)
            canvas[top : top + tile_height, left : left + tile_width] = tile

    return canvas


def write_bmp(path: Path, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("BMP output requires an HxWx3 RGB image.")

    height, width, _ = image.shape
    row_stride = (width * 3 + 3) & ~3
    pixel_data_size = row_stride * height
    file_size = 14 + 40 + pixel_data_size

    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    dib = struct.pack(
        "<IIIHHIIIIII",
        40,
        width,
        height,
        1,
        24,
        0,
        pixel_data_size,
        2835,
        2835,
        0,
        0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(dib)
        pad = b"\x00" * (row_stride - width * 3)
        for row in image[::-1]:
            handle.write(row[:, [2, 1, 0]].tobytes())
            handle.write(pad)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    use_amp = device.type == "cuda"

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.episode_index < 0 or args.episode_index >= len(manifest["episode_files"]):
        raise SystemExit(f"episode-index must be in [0, {len(manifest['episode_files']) - 1}]")

    episode = manifest["episode_files"][args.episode_index]
    episode_path = args.dataset_dir / episode["path"]
    with np.load(episode_path) as data:
        observations = data["observations"]
        actions = data["actions"].astype(np.int64)

    max_start = len(actions) - args.num_steps
    if args.start_step < 0 or args.start_step > max_start:
        raise SystemExit(f"start-step must be in [0, {max_start}]")

    vae, rnn = load_models(args.vae_checkpoint, args.rnn_checkpoint, device)

    observation_tensor = observation_batch_to_tensor(observations).to(device)
    with torch.no_grad():
        with maybe_autocast(device, use_amp):
            latents = vae.encode_mean(observation_tensor)

    hidden = rnn.init_hidden(batch_size=1, device=device)
    for step_idx in range(args.start_step):
        z_t = latents[step_idx : step_idx + 1]
        action_t = torch.tensor([int(actions[step_idx])], device=device)
        with torch.no_grad():
            _, hidden = rnn.forward_step(z_t, action_t, hidden)

    current_row: list[np.ndarray] = []
    recon_row: list[np.ndarray] = []
    true_next_row: list[np.ndarray] = []
    predicted_row: list[np.ndarray] = []
    column_metadata: list[dict[str, object]] = []

    for step_idx in range(args.start_step, args.start_step + args.num_steps):
        z_t = latents[step_idx : step_idx + 1]
        action_t = torch.tensor([int(actions[step_idx])], device=device)

        with torch.no_grad():
            with maybe_autocast(device, use_amp):
                reconstruction = vae.decode(z_t)
                outputs, hidden = rnn.forward_step(z_t, action_t, hidden)
                component_idx = int(outputs["mixture_logits"].argmax(dim=-1).item())
                predicted_z = outputs["mixture_mu"][:, component_idx, :]
                predicted_next = vae.decode(predicted_z)

        current_row.append(image_to_rgb(observations[step_idx]))
        recon_row.append(tensor_to_rgb_image(reconstruction))
        true_next_row.append(image_to_rgb(observations[step_idx + 1]))
        predicted_row.append(tensor_to_rgb_image(predicted_next))
        column_metadata.append(
            {
                "step": step_idx,
                "action": int(actions[step_idx]),
                "action_meaning": manifest["action_meanings"][int(actions[step_idx])],
                "mixture_component": component_idx,
            }
        )

    sheet = make_contact_sheet(
        [current_row, recon_row, true_next_row, predicted_row],
    )
    write_bmp(args.output_image, sheet)

    metadata_path = args.output_image.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "episode_index": args.episode_index,
                "episode_path": episode["path"],
                "start_step": args.start_step,
                "num_steps": args.num_steps,
                "row_order": ROW_ORDER,
                "columns": column_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote image: {args.output_image}")
    print(f"Wrote metadata: {metadata_path}")
    print("Row order:")
    for row_name in ROW_ORDER:
        print(f"- {row_name}")


if __name__ == "__main__":
    main()
