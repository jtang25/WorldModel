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

The collector now supports total-step targets and multiple concurrent environments, which is much more practical for large runs.

```powershell
python collect_pong_dataset.py --total-steps 1500000 --num-envs 16 --max-steps-per-episode 4096 --obs-type grayscale --resize 64 --output-dir data/pong_world_model_h200
```

The collector still uses a random policy, which is acceptable for bootstrapping the VAE and latent dynamics model but is not enough by itself to learn a strong controller.
The output directory must be empty or absent before each run so rollouts never get mixed across collections.

Useful options:

```powershell
python collect_pong_dataset.py --episodes 1000 --max-steps-per-episode 4096
python collect_pong_dataset.py --total-steps 500000 --num-envs 8
python collect_pong_dataset.py --obs-type rgb
python collect_pong_dataset.py --full-action-space
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

The upgraded stack is designed around `64 x 64` observations, a larger residual VAE, a deeper MDN-RNN, and a PPO controller that acts on frozen VAE + RNN features.

Smoke test pipeline:

```powershell
python train_world_model_pipeline.py --run-dir runs/pong_world_model_smoke_v2 --preset smoke
```

H200-oriented pipeline:

```powershell
python train_world_model_pipeline.py --run-dir runs/pong_world_model_h200 --preset h200
```

The `h200` preset is sized for real training, not quick verification. It collects about `1.5M` steps, trains a larger VAE and MDN-RNN, and runs PPO for millions of controller timesteps.

You can also run each stage separately:

```powershell
python train_vae.py --dataset-dir runs/pong_world_model_h200/dataset --output-dir runs/pong_world_model_h200/vae --latent-dim 128 --hidden-dims 96,192,384,768 --residual-blocks 2 --epochs 40 --batch-size 1024
python encode_latents.py --dataset-dir runs/pong_world_model_h200/dataset --checkpoint runs/pong_world_model_h200/vae/best.pt --output-dir runs/pong_world_model_h200/latents
python train_mdn_rnn.py --dataset-dir runs/pong_world_model_h200/latents --output-dir runs/pong_world_model_h200/mdn_rnn --hidden-size 1024 --num-layers 2 --num-mixtures 8 --epochs 30 --batch-size 512
python train_controller.py --vae-checkpoint runs/pong_world_model_h200/vae/best.pt --rnn-checkpoint runs/pong_world_model_h200/mdn_rnn/best.pt --output-dir runs/pong_world_model_h200/controller --num-envs 16 --rollout-steps 256 --total-timesteps 5000000 --hidden-dims 512,512
```

Notes:

- `train_vae.py` and `train_mdn_rnn.py` default to `--num-workers 0` so they run safely in this workspace; on a real training box you can increase that.
- The controller is now PPO, not the earlier REINFORCE baseline.
- Better frame prediction quality depends mostly on the VAE and dataset size. Better gameplay depends heavily on the PPO controller run length.

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
