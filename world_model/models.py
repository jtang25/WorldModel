from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.block(x))


class ConvVAE(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        image_size: int,
        latent_dim: int,
        hidden_dims: Sequence[int] | None = None,
        residual_blocks: int = 1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims or [64, 128, 256, 512])
        self.residual_blocks = residual_blocks

        if image_size % (2 ** len(self.hidden_dims)) != 0:
            raise ValueError(
                "image_size must be divisible by 2 ** len(hidden_dims) for the VAE."
            )

        encoder_layers: list[nn.Module] = []
        channels = input_channels
        spatial = image_size
        for hidden_dim in self.hidden_dims:
            encoder_layers.append(
                ConvNormAct(
                    channels,
                    hidden_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            for _ in range(residual_blocks):
                encoder_layers.append(ResidualBlock(hidden_dim))
            channels = hidden_dim
            spatial //= 2

        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_channels = channels
        self.encoder_spatial = spatial
        flattened_dim = channels * spatial * spatial
        self.encoder_mu = nn.Linear(flattened_dim, latent_dim)
        self.encoder_logvar = nn.Linear(flattened_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, flattened_dim)

        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(self.hidden_dims))
        current_channels = reversed_dims[0]
        for next_channels in reversed_dims[1:]:
            for _ in range(residual_blocks):
                decoder_layers.append(ResidualBlock(current_channels))
            decoder_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    ConvNormAct(current_channels, next_channels),
                )
            )
            current_channels = next_channels

        for _ in range(residual_blocks):
            decoder_layers.append(ResidualBlock(current_channels))
        decoder_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(current_channels, input_channels, kernel_size=3, padding=1),
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        hidden = torch.flatten(hidden, start_dim=1)
        return self.encoder_mu(hidden), self.encoder_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_input(z)
        hidden = hidden.view(
            -1,
            self.encoder_channels,
            self.encoder_spatial,
            self.encoder_spatial,
        )
        return self.decoder(hidden)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        reconstruction = torch.sigmoid(logits)
        return reconstruction, mu, logvar, z, logits

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return mu


def build_vae_from_config(config: dict[str, object]) -> ConvVAE:
    image_size_value = config.get("image_size", [64, 64])
    if isinstance(image_size_value, (list, tuple)):
        image_size = int(image_size_value[0])
    else:
        image_size = int(image_size_value)
    return ConvVAE(
        input_channels=int(config["input_channels"]),
        image_size=image_size,
        latent_dim=int(config["latent_dim"]),
        hidden_dims=list(config.get("hidden_dims", [64, 128, 256, 512])),
        residual_blocks=int(config.get("residual_blocks", 1)),
    )


def vae_loss(
    reconstruction: torch.Tensor,
    reconstruction_logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float,
    recon_loss_type: str = "bce",
    foreground_weight: float = 1.0,
    foreground_threshold: float = 0.05,
    free_nats: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if recon_loss_type == "bce":
        recon_per_pixel = F.binary_cross_entropy_with_logits(
            reconstruction_logits,
            target,
            reduction="none",
        )
    elif recon_loss_type == "mse":
        recon_per_pixel = F.mse_loss(reconstruction, target, reduction="none")
    elif recon_loss_type == "smooth_l1":
        recon_per_pixel = F.smooth_l1_loss(reconstruction, target, reduction="none")
    else:
        raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}")

    foreground_mask = (target > foreground_threshold).float()
    weights = 1.0 + foreground_mask * max(foreground_weight - 1.0, 0.0)
    recon = (recon_per_pixel * weights).mean()

    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )
    if free_nats > 0.0:
        kl_per_sample = torch.clamp(kl_per_sample, min=free_nats)
    kl = kl_per_sample.mean() / mu.shape[1]
    total = recon + beta * kl
    return total, recon, kl


class MDNRNN(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        action_space_n: int,
        hidden_size: int,
        num_mixtures: int,
        action_embed_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures
        self.action_embed_dim = action_embed_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.action_embedding = nn.Embedding(action_space_n, action_embed_dim)
        self.rnn = nn.LSTM(
            input_size=latent_dim + action_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.mixture_logits = nn.Linear(hidden_size, num_mixtures)
        self.mixture_mu = nn.Linear(hidden_size, num_mixtures * latent_dim)
        self.mixture_logstd = nn.Linear(hidden_size, num_mixtures * latent_dim)
        self.done_head = nn.Linear(hidden_size, 1)
        self.reward_head = nn.Linear(hidden_size, 1)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return zeros, zeros.clone()

    def _prepare_inputs(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_features = self.action_embedding(actions)
        return torch.cat([z, action_features], dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        inputs = self._prepare_inputs(z, actions)
        outputs, hidden = self.rnn(inputs, hidden)
        outputs = self.output_norm(outputs)
        batch, steps, _ = outputs.shape

        mixture_logits = self.mixture_logits(outputs)
        mixture_mu = self.mixture_mu(outputs).view(
            batch,
            steps,
            self.num_mixtures,
            self.latent_dim,
        )
        mixture_logstd = self.mixture_logstd(outputs).view(
            batch,
            steps,
            self.num_mixtures,
            self.latent_dim,
        )
        mixture_logstd = torch.clamp(mixture_logstd, min=-7.0, max=7.0)

        return (
            {
                "mixture_logits": mixture_logits,
                "mixture_mu": mixture_mu,
                "mixture_logstd": mixture_logstd,
                "done_logits": self.done_head(outputs).squeeze(-1),
                "reward": self.reward_head(outputs).squeeze(-1),
            },
            hidden,
        )

    def forward_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        outputs, hidden = self.forward(z.unsqueeze(1), action.unsqueeze(1), hidden)
        squeezed = {name: tensor[:, 0] for name, tensor in outputs.items()}
        return squeezed, hidden


def build_rnn_from_config(config: dict[str, object]) -> MDNRNN:
    return MDNRNN(
        latent_dim=int(config["latent_dim"]),
        action_space_n=int(config["action_space_n"]),
        hidden_size=int(config["hidden_size"]),
        num_mixtures=int(config["num_mixtures"]),
        action_embed_dim=int(config.get("action_embed_dim", 32)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    )


def mdn_loss(
    mixture_logits: torch.Tensor,
    mixture_mu: torch.Tensor,
    mixture_logstd: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    target = target.unsqueeze(2)
    log_mix = F.log_softmax(mixture_logits, dim=-1)
    inv_sigma = torch.exp(-mixture_logstd)
    normalized = (target - mixture_mu) * inv_sigma
    log_component = -0.5 * (
        normalized.pow(2) + 2.0 * mixture_logstd + math.log(2.0 * math.pi)
    ).sum(dim=-1)
    return -torch.logsumexp(log_mix + log_component, dim=-1).mean()


class ActorCriticController(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        action_space_n: int,
        hidden_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_space_n = action_space_n
        self.hidden_dims = list(hidden_dims or [512, 512])

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(inplace=True),
                ]
            )
            prev_dim = hidden_dim

        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.policy_head = nn.Linear(prev_dim, action_space_n)
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


def build_actor_critic_from_config(config: dict[str, object]) -> ActorCriticController:
    return ActorCriticController(
        input_dim=int(config["input_dim"]),
        action_space_n=int(config["action_space_n"]),
        hidden_dims=list(config.get("hidden_dims", [512, 512])),
    )
