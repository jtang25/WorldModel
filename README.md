# Atari Pong with Gymnasium + ALE

This workspace targets current Gymnasium 1.x with `ale-py` for Atari support and a small smoke test for `ALE/Pong-v5`.

## 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Run Pong

Headless smoke test:

```powershell
python pong_smoke_test.py
```

If PowerShell script activation is blocked, run the venv interpreter directly:

```powershell
.venv\Scripts\python.exe pong_smoke_test.py
```

Open an SDL window:

```powershell
python pong_smoke_test.py --render-mode human
```

Use the full Atari action space:

```powershell
python pong_smoke_test.py --full-action-space
```

## 4. Collect a world-model dataset

This collector saves full episodes in a format that is directly useful for the VAE -> sequence model pipeline from the World Models-style setup.

```powershell
python collect_pong_dataset.py --episodes 1000 --max-steps 4096
```

By default, it writes resized `64 x 64` RGB frames to `data/pong_world_model/`.
The collector currently uses a random policy, which is fine for bootstrapping the VAE and dynamics model but will not give you strong gameplay behavior by itself.
The output directory must be empty or absent before each run so rollouts never get mixed across collections.

Useful options:

```powershell
python collect_pong_dataset.py --obs-type grayscale
python collect_pong_dataset.py --resize 0
python collect_pong_dataset.py --full-action-space
python collect_pong_dataset.py --output-dir data/pong_world_model_gray
```

Dataset layout:

- `manifest.json`: dataset metadata, action meanings, and per-episode file list
- `episodes/episode_000000.npz`: one compressed rollout per episode

Each episode file contains:

- `observations`: shape `[T + 1, H, W, C]` for RGB or `[T + 1, H, W]` for grayscale
- `actions`: shape `[T]`
- `rewards`: shape `[T]`
- `terminated`: shape `[T]`
- `truncated`: shape `[T]`

How this maps onto world-model training:

- Train the VAE on all saved frames from `observations`
- Encode each frame into latent vectors `z_t`
- Train the recurrent dynamics model on `(z_t, action_t) -> z_{t+1}` and termination flags
- Later train the controller on top of the learned latent state and recurrent hidden state

## 5. Train the world model stack

The training scripts in this repo assume `64 x 64` observations for the VAE, so keep `--resize 64` for trainable datasets.

One-command pipeline:

```powershell
python train_world_model_pipeline.py --run-dir runs/pong_world_model
```

That will:

- collect a fresh grayscale dataset
- train a VAE on frames
- encode the dataset into latent sequences
- train the MDN-RNN on latent dynamics
- train a controller using the frozen VAE and MDN-RNN state on the real environment

You can also run each stage separately:

```powershell
python train_vae.py --dataset-dir runs/pong_world_model/dataset --output-dir runs/pong_world_model/vae
python encode_latents.py --dataset-dir runs/pong_world_model/dataset --checkpoint runs/pong_world_model/vae/best.pt --output-dir runs/pong_world_model/latents
python train_mdn_rnn.py --dataset-dir runs/pong_world_model/latents --output-dir runs/pong_world_model/mdn_rnn
python train_controller.py --vae-checkpoint runs/pong_world_model/vae/best.pt --rnn-checkpoint runs/pong_world_model/mdn_rnn/best.pt --output-dir runs/pong_world_model/controller
```

This is a baseline workflow, not a tuned Pong solution. Short runs are enough to verify the pipeline and produce checkpoints, but not enough to learn a strong policy.

## 6. Visualize imagined next frames

This script renders a contact sheet showing what the model predicts for the next frame under recorded actions.

```powershell
python visualize_imagination.py --dataset-dir runs/pong_world_model/dataset --vae-checkpoint runs/pong_world_model/vae/best.pt --rnn-checkpoint runs/pong_world_model/mdn_rnn/best.pt --output-image runs/pong_world_model/imagination.bmp
```

The output image rows are:

- current frame
- VAE reconstruction of the current frame
- true next frame
- MDN-RNN predicted next frame

## Notes

- The script registers `ale-py` with Gymnasium before creating the environment, which is required for Gymnasium 1.x projects.
- This setup uses `ALE/Pong-v5`, the current Pong environment id in the ALE namespace.
- Older Gymnasium/ALE guides may mention AutoROM or `gymnasium[accept-rom-license]`. This workspace intentionally targets the newer `gymnasium[atari]` path.
- Verified locally in this workspace with `gymnasium==1.2.3` and `ale-py==0.11.2`.
"# WorldModel" 
