import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataloader import CMUMotionDataset


class ConvBlock(nn.Module):
    """Strided temporal convolution block with ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, kernel_size: int = 3, p_drop: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return self.drop(x)


class DeconvBlock(nn.Module):
    """Temporal deconvolution block mirroring ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, p_drop: float = 0.0):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 4
        )
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.deconv(x)
        x = self.act(x)
        return self.drop(x)


class MotionAutoencoder(nn.Module):
    """Temporal convolutional autoencoder tailored for motion windows."""

    def __init__(self, input_dim: int = 63, latent_channels: int = 64, output_dim: Optional[int] = None):
        super().__init__()

        self.output_dim = output_dim if output_dim is not None else input_dim
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            ConvBlock(input_dim, 256, stride=2, kernel_size=15, p_drop=0.1),
            ConvBlock(256, 128, stride=2, kernel_size=15, p_drop=0.1),
            ConvBlock(128, self.latent_channels, stride=2, kernel_size=15, p_drop=0.0)
        )

        self.decoder = nn.Sequential(
            DeconvBlock(self.latent_channels, 128, kernel_size=4, p_drop=0.1),
            DeconvBlock(128, 256, kernel_size=4, p_drop=0.1),
            nn.ConvTranspose1d(256, self.output_dim, kernel_size=4, stride=2, padding=1)
        )

        self.output_activation = nn.Identity()
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.ConvTranspose1d):
                if module.out_channels == self.output_dim:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x):
        """Encode a time-major tensor [B, T, F] to latent representation."""
        x_channels_first = x.transpose(1, 2)
        return self.encoder(x_channels_first)

    def decode(self, z, target_length=None):
        """Decode latent tensor back to feature sequence."""
        y_channels_first = self.decoder(z)
        if target_length is not None:
            y_channels_first = self._match_time_dim(y_channels_first, target_length)
        y_channels_first = self.output_activation(y_channels_first)
        return y_channels_first.transpose(1, 2)

    def forward(self, x, corrupt_input=False, corruption_prob=0.1):
        """Forward pass with optional element dropout corruption."""
        if corrupt_input and self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - corruption_prob))
            x_used = x * mask
        else:
            x_used = x

        temporal_len = x_used.size(1)
        z = self.encode(x_used)
        x_reconstructed = self.decode(z, target_length=temporal_len)
        return x_reconstructed, z

    @staticmethod
    def _match_time_dim(tensor, target_length):
        """Trim or pad the decoded sequence so it matches the original length."""
        current = tensor.size(-1)
        if current == target_length:
            return tensor
        if current > target_length:
            return tensor[..., :target_length]
        pad_len = target_length - current
        return nn.functional.pad(tensor, (0, pad_len))


class MotionManifoldTrainer:
    """Trainer for the Motion Manifold Convolutional Autoencoder"""
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        epochs: int = 25,
        fine_tune_epochs: int = 25,
        learning_rate: float = 1e-3,
        fine_tune_lr: float = 3e-4,
        sparsity_weight: float = 0.01,
        window_size: int = 160,
        val_split: float = 0.1,
        device: str = None
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, "cache")
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.sparsity_weight = sparsity_weight
        self.window_size = window_size
        self.val_split = val_split
        self.velocity_loss_weight = 0.1
        self.latent_channels = 64
        self.velocity_mean = None
        self.velocity_std = None
        self.velocity_mean_cpu = None
        self.velocity_std_cpu = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        self._load_dataset()
        
        self._init_model()
        
    def _load_dataset(self):
        """Load the CMU Motion dataset and create training/validation splits"""
        self.dataset = CMUMotionDataset(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            frame_rate=30,
            window_size=self.window_size,
            overlap=0.5,
            include_velocity=True,
            include_foot_contact=True
        )
        
        val_size = int(self.val_split * len(self.dataset))
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Dataset loaded with {len(self.dataset)} windows from {len(self.dataset.motion_data)} files")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        self.mean_pose = torch.tensor(self.dataset.get_mean_pose(), device=self.device, dtype=torch.float32)
        self.std = torch.tensor(self.dataset.get_std(), device=self.device, dtype=torch.float32)
        self.joint_names = self.dataset.get_joint_names()
        self.joint_parents = self.dataset.get_joint_parents()
        
    def _init_model(self):
        """Initialize the motion autoencoder model"""
        sample = self.dataset[0]

        if "positions_normalized_flat" in sample:
            positions_flat = sample["positions_normalized_flat"]
        elif "positions_flat" in sample:
            positions_flat = sample["positions_flat"]
        else:
            positions_flat = sample["positions"].reshape(sample["positions"].shape[0], -1)

        self.position_feature_dim = positions_flat.shape[1]
        self.velocity_feature_dim = 0
        self.uses_velocity = False

        if "trans_vel_xz" in sample and "rot_vel_y" in sample:
            trans_vel_xz = sample["trans_vel_xz"]
            rot_velocity = sample["rot_vel_y"]
            rot_dim = rot_velocity.shape[1] if rot_velocity.dim() > 1 else 1
            self.velocity_feature_dim = trans_vel_xz.shape[1] + rot_dim
            self.uses_velocity = True
            print(
                f"Input includes positions ({self.position_feature_dim}) and velocities ({self.velocity_feature_dim})"
            )
        else:
            print(f"Input only includes positions ({self.position_feature_dim})")

        self.total_feature_dim = self.position_feature_dim + self.velocity_feature_dim

        if not self.uses_velocity:
            self.velocity_loss_weight = 0.0

        self.model = MotionAutoencoder(
            input_dim=self.total_feature_dim,
            output_dim=self.total_feature_dim,
            latent_channels=self.latent_channels
        ).to(self.device)

        if self.uses_velocity:
            self._compute_velocity_normalizer()

        print(
            f"Created model with feature dimension: {self.total_feature_dim} (positions: {self.position_feature_dim}, "
            f"velocity: {self.velocity_feature_dim})"
        )

    def _prepare_batch(self, batch):
        """Assemble the feature tensor [B, T, F] used for the autoencoder."""
        positions = batch["positions_normalized_flat"].to(self.device, dtype=torch.float32)
        features = [positions]

        if self.uses_velocity:
            trans_vel_xz = batch["trans_vel_xz"].to(self.device, dtype=torch.float32)
            rot_vel_y = batch["rot_vel_y"].to(self.device, dtype=torch.float32)
            if rot_vel_y.dim() == 2:
                rot_vel_y = rot_vel_y.unsqueeze(-1)
            velocity_features = torch.cat([trans_vel_xz, rot_vel_y], dim=-1)
            if self.velocity_mean is not None and self.velocity_std is not None:
                velocity_features = (velocity_features - self.velocity_mean) / self.velocity_std
            else:
                batch_mean = velocity_features.mean(dim=(0, 1), keepdim=True)
                batch_std = velocity_features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
                velocity_features = (velocity_features - batch_mean) / batch_std
            features.append(velocity_features.float())

        feature_tensor = torch.cat(features, dim=-1)
        return feature_tensor.to(self.device, dtype=torch.float32)

    def _compute_velocity_normalizer(self):
        """Estimate dataset-level mean and std for velocity features."""

        loader = DataLoader(
            self.dataset,
            batch_size=min(256, self.batch_size * 2),
            shuffle=True,
            num_workers=0
        )

        total = 0
        mean = torch.zeros(self.velocity_feature_dim, dtype=torch.float64)
        squared = torch.zeros_like(mean)

        for _, batch in enumerate(loader):
            trans = batch["trans_vel_xz"].double()
            rot = batch["rot_vel_y"].double()
            if rot.dim() == 2:
                rot = rot.unsqueeze(-1)
            velocities = torch.cat([trans, rot], dim=-1)
            mean += velocities.sum(dim=(0, 1))
            squared += (velocities ** 2).sum(dim=(0, 1))
            total += velocities.shape[0] * velocities.shape[1]

            if total >= 20000:  # sample a subset to keep preprocessing quick
                break

        if total == 0:
            self.velocity_mean = torch.zeros(self.velocity_feature_dim, device=self.device, dtype=torch.float32).view(1, 1, -1)
            self.velocity_std = torch.ones(self.velocity_feature_dim, device=self.device, dtype=torch.float32).view(1, 1, -1)
            self.velocity_mean_cpu = self.velocity_mean.cpu().clone()
            self.velocity_std_cpu = self.velocity_std.cpu().clone()
            return
        else:
            mean /= total
            var = squared / total - mean ** 2
            std = torch.sqrt(var.clamp(min=1e-6))

            self.velocity_mean = mean.to(self.device, dtype=torch.float32).view(1, 1, -1)
            self.velocity_std = std.to(self.device, dtype=torch.float32).view(1, 1, -1)
            self.velocity_mean_cpu = self.velocity_mean.detach().cpu().clone()
            self.velocity_std_cpu = self.velocity_std.detach().cpu().clone()
        
    def train(self):
        """Train the motion autoencoder in two phases: initial training and fine-tuning"""
        
        phase1_stats = self._train_phase(
            self.epochs,
            self.learning_rate,
            0.1,
            self.sparsity_weight,
            "phase1_initial"
        )

        phase2_stats = self._train_phase(
            self.fine_tune_epochs,
            self.fine_tune_lr,
            0.0,
            self.sparsity_weight * 0.5,
            "phase2_finetune"
        )
        
        all_stats = {
            "phase1_initial": phase1_stats,
            "phase2_finetune": phase2_stats
        }
        
        with open(os.path.join(self.output_dir, "training_stats.json"), "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2)
            
        self._save_model()
        self._save_normalization_params()
        self._plot_training_curves(all_stats)
        
        return all_stats

    def _train_phase(self, epochs, learning_rate, corruption_prob, sparsity_weight, phase_name):
        """Train the model for a specific phase (initial training or fine-tuning)"""
        print(f"\n===== {phase_name.capitalize()} Training Phase =====")

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler_gamma = 0.9 if phase_name == "phase1_initial" else 0.95
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
        criterion = nn.MSELoss()

        stats = {
            "train_total_loss": [],
            "train_position_loss": [],
            "train_velocity_loss": [],
            "train_sparsity_loss": [],
            "val_total_loss": [],
            "val_position_loss": [],
            "val_velocity_loss": []
        }

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()

            train_position_sum = 0.0
            train_velocity_sum = 0.0
            train_sparsity_sum = 0.0
            train_total_sum = 0.0
            train_samples = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in progress_bar:
                inputs = self._prepare_batch(batch)

                optimizer.zero_grad()

                recon, latent = self.model(
                    inputs,
                    corrupt_input=(corruption_prob > 0.0),
                    corruption_prob=corruption_prob
                )

                position_loss = criterion(
                    recon[..., :self.position_feature_dim],
                    inputs[..., :self.position_feature_dim]
                )

                velocity_loss = torch.tensor(0.0, device=self.device)
                if self.velocity_feature_dim > 0:
                    velocity_slice = slice(
                        self.position_feature_dim,
                        self.position_feature_dim + self.velocity_feature_dim
                    )
                    velocity_loss = criterion(recon[..., velocity_slice], inputs[..., velocity_slice])

                sparsity_loss = torch.tensor(0.0, device=self.device)
                if sparsity_weight > 0:
                    sparsity_loss = latent.abs().mean()

                loss = (
                    position_loss
                    + self.velocity_loss_weight * velocity_loss
                    + sparsity_weight * sparsity_loss
                )

                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size = inputs.size(0)
                train_samples += batch_size
                train_position_sum += position_loss.item() * batch_size
                train_velocity_sum += velocity_loss.item() * batch_size
                train_sparsity_sum += sparsity_loss.item() * batch_size
                train_total_sum += loss.item() * batch_size

                progress_bar.set_postfix({
                    "total": f"{loss.item():.4f}",
                    "pos": f"{position_loss.item():.4f}",
                    "vel": f"{velocity_loss.item():.4f}"
                })

            mean_train_position = train_position_sum / max(1, train_samples)
            mean_train_velocity = train_velocity_sum / max(1, train_samples)
            mean_train_sparsity = train_sparsity_sum / max(1, train_samples)
            mean_train_total = train_total_sum / max(1, train_samples)

            stats["train_total_loss"].append(mean_train_total)
            stats["train_position_loss"].append(mean_train_position)
            stats["train_velocity_loss"].append(mean_train_velocity)
            stats["train_sparsity_loss"].append(mean_train_sparsity)

            self.model.eval()
            val_position_sum = 0.0
            val_velocity_sum = 0.0
            val_total_sum = 0.0
            val_samples = 0

            with torch.no_grad():
                progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch in progress_bar:
                    inputs = self._prepare_batch(batch)
                    recon, latent = self.model(inputs, corrupt_input=False)

                    position_loss = criterion(
                        recon[..., :self.position_feature_dim],
                        inputs[..., :self.position_feature_dim]
                    )

                    velocity_loss = torch.tensor(0.0, device=self.device)
                    if self.velocity_feature_dim > 0:
                        velocity_slice = slice(
                            self.position_feature_dim,
                            self.position_feature_dim + self.velocity_feature_dim
                        )
                        velocity_loss = criterion(recon[..., velocity_slice], inputs[..., velocity_slice])

                    total_loss = position_loss + self.velocity_loss_weight * velocity_loss

                    batch_size = inputs.size(0)
                    val_samples += batch_size
                    val_position_sum += position_loss.item() * batch_size
                    val_velocity_sum += velocity_loss.item() * batch_size
                    val_total_sum += total_loss.item() * batch_size

                    progress_bar.set_postfix({
                        "total": f"{total_loss.item():.4f}",
                        "pos": f"{position_loss.item():.4f}"
                    })

            mean_val_position = val_position_sum / max(1, val_samples)
            mean_val_velocity = val_velocity_sum / max(1, val_samples)
            mean_val_total = val_total_sum / max(1, val_samples)

            stats["val_total_loss"].append(mean_val_total)
            stats["val_position_loss"].append(mean_val_position)
            stats["val_velocity_loss"].append(mean_val_velocity)

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch+1}/{epochs} - Train Total: {mean_train_total:.4f}, "
                f"Val Total: {mean_val_total:.4f}, LR: {current_lr:.5f}"
            )

            if mean_val_total < best_val_loss:
                best_val_loss = mean_val_total
                self._save_checkpoint(epoch, end=f'_valloss_{mean_val_total:.6f}', phase_name=phase_name)
                print(f"  Saved checkpoint with val_loss: {mean_val_total:.6f}")

            scheduler.step()

        return stats
    
    def _save_checkpoint(self, epoch, end, phase_name):
        """Save a model checkpoint"""
        checkpoint_path = os.path.join(
            self.output_dir, "checkpoints", f"{phase_name}_epoch_{epoch+1}{end}.pt"
        )
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
        }, checkpoint_path)
    
    def _save_model(self):
        """Save the trained model"""
        model_path = os.path.join(self.output_dir, "models", "motion_autoencoder.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def _save_normalization_params(self):
        """Save normalization parameters for inference if needed"""
        norm_data = {
            "mean_pose": self.mean_pose.cpu().numpy(),
            "std": self.std.cpu().numpy(),
            "joint_names": self.joint_names,
            "joint_parents": self.joint_parents,
            "position_feature_dim": self.position_feature_dim,
            "velocity_feature_dim": self.velocity_feature_dim,
            "total_feature_dim": self.total_feature_dim,
            "velocity_loss_weight": self.velocity_loss_weight
        }

        if self.velocity_feature_dim > 0 and self.velocity_mean_cpu is not None and self.velocity_std_cpu is not None:
            norm_data["velocity_mean"] = self.velocity_mean_cpu.numpy()
            norm_data["velocity_std"] = self.velocity_std_cpu.numpy()
        
        np.save(os.path.join(self.output_dir, "normalization.npy"), norm_data)
        print(f"Normalization parameters saved to {self.output_dir}/normalization.npy")
    
    def _plot_training_curves(self, stats):
        """Plot training curves for one or more training phases"""
        if not isinstance(stats[list(stats.keys())[0]], dict):
            stats = {"train": stats}
            
        n_p = len(list(stats.keys()))
        plt.figure(figsize=(12, 4 * n_p))
        for i, (phase_name, phase_stats) in enumerate(stats.items()):
            plt.subplot(n_p, 1, i+1)
            for key, values in phase_stats.items():
                plt.plot(values, label=key)
            plt.title(f"{phase_name.capitalize()} Training Phase")
            plt.xlabel("Epoch")
            plt.ylabel("Statistics")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "training_curves.png"))
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/plots/training_curves.png")


class MotionManifoldSynthesizer:
    """Synthesizer for generating, fixing, and analyzing motion using the learned manifold"""
    def __init__(
        self,
        model_path: str,
        dataset: CMUMotionDataset,
        device: str = None
    ):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        self.latent_channels = 96
        
        self._load_normalization(dataset)
        self._configure_feature_layout(dataset)
        
        self._load_model(model_path)
    
    def _load_normalization(self, dataset: CMUMotionDataset):
        """Load normalization parameters from dataset"""
        self.mean_pose = torch.tensor(dataset.mean_pose, device=self.device, dtype=torch.float32)
        self.std = torch.tensor(dataset.std, device=self.device, dtype=torch.float32)
        self.joint_names = dataset.joint_names
        self.joint_parents = dataset.joint_parents

    def _configure_feature_layout(self, dataset: CMUMotionDataset):
        """Determine how many features were used for training."""

        sample = dataset[0]

        if "positions_normalized_flat" in sample:
            positions_flat = sample["positions_normalized_flat"]
        elif "positions_flat" in sample:
            positions_flat = sample["positions_flat"]
        else:
            positions_flat = sample["positions"].reshape(sample["positions"].shape[0], -1)

        self.position_feature_dim = positions_flat.shape[1]

        self.velocity_feature_dim = 0
        if "trans_vel_xz" in sample and "rot_vel_y" in sample:
            self.velocity_feature_dim = sample["trans_vel_xz"].shape[1] + 1

        self.total_feature_dim = self.position_feature_dim + self.velocity_feature_dim
    
    def _load_model(self, model_path):
        """Load trained model"""
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location=self.device)

            input_dim = None
            latent_channels = None
            output_dim = None

            for key, value in model_state.items():
                if key.endswith("encoder.0.conv.weight"):
                    input_dim = value.shape[1]
                elif key.endswith("encoder.2.conv.weight"):
                    latent_channels = value.shape[0]
                elif key.endswith("decoder.2.weight"):
                    output_dim = value.shape[1]

            if input_dim is None:
                input_dim = self.position_feature_dim
            if output_dim is None:
                output_dim = input_dim
            if latent_channels is None:
                latent_channels = self.latent_channels if hasattr(self, "latent_channels") else 96

            self.total_feature_dim = input_dim
            self.velocity_feature_dim = max(0, self.total_feature_dim - self.position_feature_dim)
            self.latent_channels = latent_channels

            self.model = MotionAutoencoder(
                input_dim=input_dim,
                output_dim=output_dim,
                latent_channels=latent_channels
            ).to(self.device)

            self.model.load_state_dict(model_state)
            self.model.eval()

            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def _assemble_features(self, normalized_positions, trans_vel=None, rot_vel=None):
        """Create feature tensor consistent with training-time ordering."""

        features = [normalized_positions]

        if self.velocity_feature_dim > 0:
            if trans_vel is None or rot_vel is None:
                velocity = torch.zeros(
                    normalized_positions.size(0),
                    normalized_positions.size(1),
                    self.velocity_feature_dim,
                    device=normalized_positions.device
                )
            else:
                if rot_vel.dim() == 2:
                    rot_vel = rot_vel.unsqueeze(-1)
                velocity = torch.cat([trans_vel, rot_vel], dim=-1)

            features.append(velocity)

        feature_tensor = torch.cat(features, dim=-1)

        if feature_tensor.size(-1) < self.total_feature_dim:
            pad = torch.zeros(
                feature_tensor.size(0),
                feature_tensor.size(1),
                self.total_feature_dim - feature_tensor.size(-1),
                device=feature_tensor.device
            )
            feature_tensor = torch.cat([feature_tensor, pad], dim=-1)
        elif feature_tensor.size(-1) > self.total_feature_dim:
            feature_tensor = feature_tensor[..., :self.total_feature_dim]

        return feature_tensor
    
    def _get_pose_stats(self, joints, dims):
        mean_pose = self.mean_pose.to(self.device).view(1, 1, joints, dims)
        std_pose = (self.std.to(self.device) + 1e-8).view(1, 1, joints, dims)
        return mean_pose, std_pose

    def _normalize_positions(self, positions):
        batch_size, time_steps, joints, dims = positions.shape
        mean_pose, std_pose = self._get_pose_stats(joints, dims)
        normalized = ((positions - mean_pose) / std_pose).view(batch_size, time_steps, -1)
        return normalized, mean_pose, std_pose

    @staticmethod
    def _denormalize_positions(flat_positions, mean_pose, std_pose, joints, dims):
        batch_size, time_steps, _ = flat_positions.shape
        reshaped = flat_positions.view(batch_size, time_steps, joints, dims)
        return reshaped * std_pose + mean_pose

    def fix_corrupted_motion(self, motion, corruption_type='zero', corruption_params=None):
        """
        Fix corrupted motion by projecting onto the manifold and recovering global motion
        
        Args:
            motion: tensor of shape [batch_size, time_steps, joints, dims]
            corruption_type: Type of corruption to apply ('zero', 'noise', or 'missing')
            corruption_params: Parameters for corruption
                    
        Returns:
            Tuple of (corrupted_motion, fixed_motion)
        """
        positions = motion['positions'].to(self.device)
        batch_size, time_steps, joints, dims = positions.shape

        if corruption_params is not None:
            corrupted_motion = self._apply_corruption(positions, corruption_type, corruption_params)
        else:
            corrupted_motion = positions.clone()

        mean_pose = self.mean_pose.to(self.device).view(1, 1, joints, dims)
        std = (self.std.to(self.device) + 1e-8).view(1, 1, joints, dims)

        normalized_positions = ((corrupted_motion - mean_pose) / std).view(batch_size, time_steps, -1)

        trans_vel = motion.get('trans_vel_xz')
        rot_vel = motion.get('rot_vel_y')
        if trans_vel is not None:
            trans_vel = trans_vel.to(self.device)
        if rot_vel is not None:
            rot_vel = rot_vel.to(self.device)

        feature_tensor = self._assemble_features(normalized_positions, trans_vel, rot_vel)

        with torch.no_grad():
            reconstructed, _ = self.model(feature_tensor, corrupt_input=False)

        recon_positions = reconstructed[..., :self.position_feature_dim].view(batch_size, time_steps, joints, dims)
        fixed_motion = recon_positions * std + mean_pose

        from dataloader import recover_global_motion
        if trans_vel is not None and rot_vel is not None:
            corrupted_global = recover_global_motion(corrupted_motion, trans_vel, rot_vel)
            fixed_global = recover_global_motion(fixed_motion, trans_vel, rot_vel)
        else:
            corrupted_global = corrupted_motion
            fixed_global = fixed_motion

        return corrupted_global, fixed_global
    
    def _apply_corruption(self, motion, corruption_type, params):
        """Apply corruption to motion data"""
        corrupted = motion.clone()
        
        if corruption_type == 'zero':
            prob = params.get('prob', 0.5)
            mask = torch.bernoulli(torch.ones_like(corrupted) * (1 - prob))
            corrupted = corrupted * mask
            
        elif corruption_type == 'noise':
            noise_scale = params.get('scale', 0.1)
            noise = torch.randn_like(corrupted) * noise_scale
            corrupted = corrupted + noise
            
        elif corruption_type == 'missing':
            joint_idx = params.get('joint_idx', 0)
            corrupted[:, :, joint_idx, :] = 0.0
            
        return corrupted
    
    def interpolate_motions(self, motion1, motion2, t):
        """
        Interpolate between two motions on the manifold, handling global transforms
        
        Args:
            motion1: tensor of shape [batch_size, time_steps, joints, dims]
            motion2: tensor of shape [batch_size, time_steps, joints, dims]
            t: Interpolation parameter (0 to 1)
                    
        Returns:
            Interpolated motion as tensor of shape [batch_size, time_steps, joints, dims]
        """
        pos1 = motion1['positions'].to(self.device)
        pos2 = motion2['positions'].to(self.device)

        batch_size, time_steps, joints, dims = pos1.shape

        mean_pose = self.mean_pose.to(self.device).view(1, 1, joints, dims)
        std = (self.std.to(self.device) + 1e-8).view(1, 1, joints, dims)

        pos1_norm = ((pos1 - mean_pose) / std).view(batch_size, time_steps, -1)
        pos2_norm = ((pos2 - mean_pose) / std).view(batch_size, time_steps, -1)

        trans_vel1 = motion1.get('trans_vel_xz')
        rot_vel1 = motion1.get('rot_vel_y')
        trans_vel2 = motion2.get('trans_vel_xz')
        rot_vel2 = motion2.get('rot_vel_y')

        if trans_vel1 is not None:
            trans_vel1 = trans_vel1.to(self.device)
        if rot_vel1 is not None:
            rot_vel1 = rot_vel1.to(self.device)
        if trans_vel2 is not None:
            trans_vel2 = trans_vel2.to(self.device)
        if rot_vel2 is not None:
            rot_vel2 = rot_vel2.to(self.device)

        feat1 = self._assemble_features(pos1_norm, trans_vel1, rot_vel1)
        feat2 = self._assemble_features(pos2_norm, trans_vel2, rot_vel2)

        with torch.no_grad():
            z1 = self.model.encode(feat1)
            z2 = self.model.encode(feat2)

        z_interp = (1 - t) * z1 + t * z2

        with torch.no_grad():
            interp_feat = self.model.decode(z_interp, target_length=time_steps)

        interp_positions = interp_feat[..., :self.position_feature_dim].view(batch_size, time_steps, joints, dims)
        interp_motion = interp_positions * std + mean_pose

        if trans_vel1 is not None and trans_vel2 is not None and rot_vel1 is not None and rot_vel2 is not None:
            trans_vel_interp = (1 - t) * trans_vel1 + t * trans_vel2
            rot_vel_interp = (1 - t) * rot_vel1 + t * rot_vel2

            from dataloader import recover_global_motion
            interp_global = recover_global_motion(interp_motion, trans_vel_interp, rot_vel_interp)
            return interp_global

        return interp_motion
    
    def complete_motion(self, motion, mask, blend=1.0):
        """Inpaint masked frames using the learned manifold."""
        positions = motion['positions'].to(self.device)
        if positions.dim() == 3:
            positions = positions.unsqueeze(0)

        batch_size, time_steps, joints, dims = positions.shape

        trans_vel = motion.get('trans_vel_xz')
        rot_vel = motion.get('rot_vel_y')

        if trans_vel is not None:
            # Accept [T, 2] or [B, T, 2], convert to [B, T, 2].
            if trans_vel.dim() == 2:
                trans_vel = trans_vel.unsqueeze(0)
            elif trans_vel.dim() != 3:
                raise ValueError(f"Unexpected trans_vel_xz shape: {tuple(trans_vel.shape)}")

        if rot_vel is not None:
            # Accept [T], [B, T], [T, 1], [B, T, 1], and [B, 1, T].
            if rot_vel.dim() == 1:
                rot_vel = rot_vel.unsqueeze(0).unsqueeze(-1)
            elif rot_vel.dim() == 2:
                if rot_vel.shape[1] == 1 and rot_vel.shape[0] == time_steps:
                    rot_vel = rot_vel.unsqueeze(0)
                else:
                    rot_vel = rot_vel.unsqueeze(-1)
            elif rot_vel.dim() == 3:
                if rot_vel.shape[1] == 1 and rot_vel.shape[2] == time_steps:
                    rot_vel = rot_vel.transpose(1, 2)
            else:
                raise ValueError(f"Unexpected rot_vel_y shape: {tuple(rot_vel.shape)}")

        if trans_vel is not None:
            if trans_vel.shape[0] == 1 and batch_size > 1:
                trans_vel = trans_vel.expand(batch_size, -1, -1)
            if trans_vel.shape[0] != batch_size or trans_vel.shape[1] != time_steps:
                raise ValueError(
                    f"trans_vel_xz must be [B, T, 2] with B={batch_size}, T={time_steps}; "
                    f"got {tuple(trans_vel.shape)}"
                )

        if rot_vel is not None:
            if rot_vel.shape[0] == 1 and batch_size > 1:
                rot_vel = rot_vel.expand(batch_size, -1, -1)
            if rot_vel.shape[0] != batch_size or rot_vel.shape[1] != time_steps:
                raise ValueError(
                    f"rot_vel_y must align to [B, T, 1] with B={batch_size}, T={time_steps}; "
                    f"got {tuple(rot_vel.shape)}"
                )

        mask = mask.to(self.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).unsqueeze(-1)
        mask = mask.clamp(0.0, 1.0)

        mean_pose, std_pose = self._get_pose_stats(joints, dims)

        corrupted_positions = mask * positions + (1.0 - mask) * mean_pose
        normalized, _, _ = self._normalize_positions(corrupted_positions)

        if trans_vel is not None:
            trans_vel = trans_vel.to(self.device)
        if rot_vel is not None:
            rot_vel = rot_vel.to(self.device)

        features = self._assemble_features(normalized, trans_vel, rot_vel)

        with torch.no_grad():
            recon_features, _ = self.model(features)

        recon_flat = recon_features[..., :self.position_feature_dim]
        recon_positions = self._denormalize_positions(recon_flat, mean_pose, std_pose, joints, dims)

        blend = float(blend)
        completed_positions = mask * positions + (1.0 - mask) * (
            blend * recon_positions + (1.0 - blend) * corrupted_positions
        )

        from dataloader import recover_global_motion

        def _recover(seq):
            if trans_vel is not None and rot_vel is not None:
                return recover_global_motion(seq, trans_vel, rot_vel)
            return seq

        corrupted_global = _recover(corrupted_positions).cpu()
        recon_global = _recover(recon_positions).cpu()
        completed_global = _recover(completed_positions).cpu()

        return {
            "corrupted": corrupted_global,
            "completed": completed_global,
            "reconstruction": recon_global,
            "mask": mask.cpu(),
        }

    def style_transfer(self, content_motion, style_motion, alpha=0.5):
        """Transfer style statistics in the latent space while preserving content structure."""
        content_positions = content_motion['positions'].to(self.device)
        style_positions = style_motion['positions'].to(self.device)

        if content_positions.dim() == 3:
            content_positions = content_positions.unsqueeze(0)
        if style_positions.dim() == 3:
            style_positions = style_positions.unsqueeze(0)

        content_trans = content_motion.get('trans_vel_xz')
        content_rot = content_motion.get('rot_vel_y')
        style_trans = style_motion.get('trans_vel_xz')
        style_rot = style_motion.get('rot_vel_y')

        min_frames = min(content_positions.size(1), style_positions.size(1))
        content_positions = content_positions[:, :min_frames]
        style_positions = style_positions[:, :min_frames]

        def _prepare_velocity_pair(trans, rot, batch_size, time_steps, name):
            if trans is not None:
                # Accept [T, 2] or [B, T, 2], convert to [B, T, 2].
                if trans.dim() == 2 and trans.shape[-1] == 2:
                    trans = trans.unsqueeze(0)
                elif trans.dim() != 3:
                    raise ValueError(f"Unexpected {name}.trans_vel_xz shape: {tuple(trans.shape)}")

                if trans.shape[0] == 1 and batch_size > 1:
                    trans = trans.expand(batch_size, -1, -1)
                if trans.shape[0] != batch_size:
                    raise ValueError(
                        f"{name}.trans_vel_xz must have batch size {batch_size}; got {tuple(trans.shape)}"
                    )
                if trans.shape[1] < time_steps:
                    raise ValueError(
                        f"{name}.trans_vel_xz has too few frames: expected at least {time_steps}, got {trans.shape[1]}"
                    )
                trans = trans[:, :time_steps]

            if rot is not None:
                # Accept [T], [B, T], [T, 1], [B, T, 1], and [B, 1, T].
                if rot.dim() == 1:
                    rot = rot.unsqueeze(0).unsqueeze(-1)
                elif rot.dim() == 2:
                    if rot.shape[1] == 1 and rot.shape[0] == time_steps:
                        rot = rot.unsqueeze(0)
                    else:
                        rot = rot.unsqueeze(-1)
                elif rot.dim() == 3:
                    if rot.shape[1] == 1 and rot.shape[2] >= time_steps:
                        rot = rot.transpose(1, 2)
                else:
                    raise ValueError(f"Unexpected {name}.rot_vel_y shape: {tuple(rot.shape)}")

                if rot.shape[0] == 1 and batch_size > 1:
                    rot = rot.expand(batch_size, -1, -1)
                if rot.shape[0] != batch_size:
                    raise ValueError(
                        f"{name}.rot_vel_y must have batch size {batch_size}; got {tuple(rot.shape)}"
                    )
                if rot.shape[1] < time_steps:
                    raise ValueError(
                        f"{name}.rot_vel_y has too few frames: expected at least {time_steps}, got {rot.shape[1]}"
                    )
                rot = rot[:, :time_steps]

            return trans, rot

        content_trans, content_rot = _prepare_velocity_pair(
            content_trans, content_rot, content_positions.size(0), min_frames, "content"
        )
        style_trans, style_rot = _prepare_velocity_pair(
            style_trans, style_rot, style_positions.size(0), min_frames, "style"
        )

        content_norm, content_mean, content_std = self._normalize_positions(content_positions)
        style_norm, _, _ = self._normalize_positions(style_positions)

        content_features = self._assemble_features(content_norm, content_trans, content_rot)
        style_features = self._assemble_features(style_norm, style_trans, style_rot)

        with torch.no_grad():
            content_latent = self.model.encode(content_features)
            style_latent = self.model.encode(style_features)

        content_mean_lat = content_latent.mean(dim=2, keepdim=True)
        content_std_lat = content_latent.std(dim=2, keepdim=True).clamp(min=1e-4)
        style_mean_lat = style_latent.mean(dim=2, keepdim=True)
        style_std_lat = style_latent.std(dim=2, keepdim=True)

        standardized = (content_latent - content_mean_lat) / content_std_lat
        styled_latent = standardized * style_std_lat + style_mean_lat
        alpha = float(alpha)
        mixed_latent = alpha * content_latent + (1.0 - alpha) * styled_latent

        with torch.no_grad():
            stylized_features = self.model.decode(mixed_latent, target_length=min_frames)

        stylized_flat = stylized_features[..., :self.position_feature_dim]
        stylized_positions = self._denormalize_positions(
            stylized_flat,
            content_mean[:, :min_frames],
            content_std[:, :min_frames],
            content_positions.size(2),
            content_positions.size(3)
        )

        from dataloader import recover_global_motion

        def _recover(seq, trans, rot):
            if trans is not None and rot is not None:
                return recover_global_motion(seq, trans, rot)
            return seq

        stylized_global = _recover(stylized_positions, content_trans, content_rot).cpu()
        content_global = _recover(content_positions, content_trans, content_rot).cpu()
        style_global = _recover(style_positions, style_trans, style_rot).cpu()

        return {
            "stylized": stylized_global,
            "content": content_global,
            "style": style_global,
        }
    
def main():
    """Example usage of the motion manifold training"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "cmu-mocap")
    output_dir = os.path.join(script_dir, "output", "ae")
    
    trainer = MotionManifoldTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=32,
        epochs=25,
        fine_tune_epochs=25,
        learning_rate=1e-3,
        fine_tune_lr=3e-4,
        sparsity_weight=0.01,
        window_size=160,
        val_split=0.1
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
    
    
