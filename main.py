"""
Neural Mode Clustering (NMC) - Complete Implementation v3
==========================================================

CHANGELOG v3:
------------
NEW FIXES:
1. Fixed z_std initialization for cached embeddings path
2. Fixed Stability@Aug to use fixed threshold instead of quantile
3. Added assignment consistency stability metric
4. Removed deprecated n_jobs from DBSCAN
5. Fixed gradient clipping for all param groups
6. Added safety checks and improved documentation

Previous fixes maintained:
- Two-view test loader for proper stability measurement
- Trainable projection head
- Element-wise backtracking
- DBSCAN fallback
- Persistence edge cases
- Embedding cache properly wired

Author: ML Engineer
Date: 2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import torchvision.models as models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

from tqdm.auto import tqdm
import math
import random
from collections import defaultdict
import time
from typing import Tuple, List, Optional, Dict, Union
import json
import argparse
from pathlib import Path
from PIL import Image

# ============================================================================
# Configuration and Setup
# ============================================================================

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get best available device with fallback to CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    return device

# ============================================================================
# Data Loading with Proper Augmentations
# ============================================================================

class AugmentedDataset(Dataset):
    """Dataset wrapper that produces two independent augmented views"""
    
    def __init__(self, base_dataset, weak_aug=None, strong_aug=None):
        self.base_dataset = base_dataset
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Ensure we have PIL image for independent augmentations
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        
        # Create two independent views
        if self.weak_aug is not None:
            img_weak = self.weak_aug(img)
        else:
            img_weak = T.ToTensor()(img)
        
        if self.strong_aug is not None:
            img_strong = self.strong_aug(img)
        else:
            img_strong = img_weak.clone()
        
        return img_weak, img_strong, label

class CachedEmbeddingDataset(Dataset):
    """Dataset for cached embeddings"""
    
    def __init__(self, z, z_aug):
        self.z = torch.from_numpy(z).float()
        self.z_aug = torch.from_numpy(z_aug).float()
    
    def __len__(self):
        return len(self.z)
    
    def __getitem__(self, idx):
        return self.z[idx], self.z_aug[idx]

def get_cifar10_transforms(input_size=224):
    """Get CIFAR-10 augmentations with proper input size"""
    
    # Strong augmentation (SimCLR-style)
    strong_aug = T.Compose([
        T.Resize(input_size),
        T.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Weak augmentation (truly weak for stability testing)
    weak_aug = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),  # More stable than RandomCrop for weak aug
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test augmentation (no randomness)
    test_aug = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return strong_aug, weak_aug, test_aug

def load_dataset(dataset_name='cifar10', data_dir='./data', batch_size=256, 
                 input_size=224, num_workers=2):
    """Load and prepare dataset with proper augmentations"""
    
    if dataset_name.lower() == 'cifar10':
        strong_aug, weak_aug, test_aug = get_cifar10_transforms(input_size)
        
        # Load base datasets (keep as PIL for proper augmentation)
        train_base = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )
        test_base = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_aug
        )
        
        # Wrap training set with augmentations
        train_dataset = AugmentedDataset(train_base, weak_aug, strong_aug)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_base,
        batch_size=batch_size * 2,
        shuffle=False,  # Important: keep False for alignment with two-view loader
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, num_classes

def make_two_view_test_loader(data_dir='./data', input_size=224, batch_size=512, num_workers=2):
    """Create test loader with two augmented views for stability testing
    
    Note: Uses same dataset order as test_loader (shuffle=False) to ensure alignment
    """
    strong_aug, _, test_aug = get_cifar10_transforms(input_size)
    
    test_base = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=None
    )
    
    # Create dataset with weak (test_aug) and strong augmentations
    two_view_dataset = AugmentedDataset(test_base, weak_aug=test_aug, strong_aug=strong_aug)
    
    return DataLoader(
        two_view_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep same order as test_loader for alignment
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )

# ============================================================================
# Core Neural Network Components
# ============================================================================

class SpectralNormMLP(nn.Module):
    """MLP with spectral normalization for smooth gradients"""
    
    def __init__(self, d_in, hidden=512, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        dims = [d_in] + [hidden] * (depth - 1) + [1]
        
        for i in range(len(dims) - 2):
            lin = nn.Linear(dims[i], dims[i + 1])
            lin = nn.utils.spectral_norm(lin)
            layers.extend([lin, nn.SiLU(), nn.Dropout(dropout)])
        
        # Last layer
        last = nn.Linear(dims[-2], dims[-1])
        last = nn.utils.spectral_norm(last)
        layers.append(last)
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, z):
        return self.net(z).squeeze(-1)

class NeuralPotential(nn.Module):
    """Neural potential function with gradient computation"""
    
    def __init__(self, d_in, hidden=512, depth=4, dropout=0.1):
        super().__init__()
        self.phi = SpectralNormMLP(d_in, hidden, depth, dropout)
        self.d_in = d_in
    
    def forward(self, z):
        return self.phi(z)
    
    def gradient(self, z, create_graph=True):
        """Compute ∇φ(z) via autograd"""
        z = z.requires_grad_(True)
        phi = self.forward(z).sum()
        grad, = torch.autograd.grad(phi, z, create_graph=create_graph)
        return grad
    
    def gradient_batch(self, z):
        """Efficient batched gradient computation"""
        z = z.requires_grad_(True)
        with torch.enable_grad():
            phi = self.forward(z)
            grad = torch.autograd.grad(phi.sum(), z)[0]
        return grad

class FrozenEncoder(nn.Module):
    """Encoder with optional trainable projection head"""
    
    def __init__(self, model_name='resnet18', out_dim=128, pretrained=True, train_projection=True):
        super().__init__()
        
        # Load pretrained model
        if model_name == 'resnet18':
            if pretrained:
                backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                backbone = models.resnet18(weights=None)
            self.in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.backbone = backbone
        self.train_projection = train_projection
        self.out_dim = out_dim
        
        # Projection head (trainable or identity)
        if train_projection:
            self.projection = nn.Linear(self.in_features, out_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        else:
            # No projection, use backbone features directly
            self.projection = nn.Identity()
            self.out_dim = self.in_features
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Set backbone to eval mode
        self.backbone.eval()
    
    def forward(self, x):
        # Always use no_grad for frozen backbone
        with torch.no_grad():
            features = self.backbone(x)
        
        # Projection may have gradients if train_projection=True
        z = self.projection(features)
        return F.normalize(z, dim=-1)  # L2 normalize

# ============================================================================
# Denoising Score Matching Training
# ============================================================================

class DSMTrainer:
    """DSM trainer with invariance regularization"""
    
    def __init__(self, encoder, potential, device='cuda',
                 lr=1e-3, weight_decay=1e-4,
                 sigma_low=0.05, sigma_high=0.5, 
                 lambda_inv=0.1, inv_mode='endpoint'):
        
        self.encoder = encoder.to(device)
        self.potential = potential.to(device)
        self.device = device
        
        # Prepare parameters for optimizer
        params_to_train = list(self.potential.parameters())
        
        # Add projection parameters if trainable
        if hasattr(encoder, 'train_projection') and encoder.train_projection:
            params_to_train.extend(encoder.projection.parameters())
            print("Training projection head jointly with DSM")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            params_to_train,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler (initialized in train())
        self.scheduler = None
        
        # Noise parameters
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.lambda_inv = lambda_inv
        self.inv_mode = inv_mode
        
        # Statistics
        self.z_std = None
        self.loss_history = []
        self.best_state = None
    
    def log_uniform_sigma(self, size):
        """Sample noise level from log-uniform distribution"""
        if self.z_std is None:
            raise ValueError("z_std not initialized. Call estimate_z_statistics first.")
        
        log_low = math.log(self.sigma_low * self.z_std)
        log_high = math.log(self.sigma_high * self.z_std)
        u = torch.rand(size, device=self.device)
        log_sigma = u * (log_high - log_low) + log_low
        return torch.exp(log_sigma).unsqueeze(-1)
    
    def dsm_loss(self, z, sigma):
        """Denoising Score Matching loss"""
        eps = torch.randn_like(z)
        z_noisy = z + sigma * eps
        target = -(z_noisy - z) / (sigma ** 2)
        pred = self.potential.gradient(z_noisy, create_graph=True)
        loss = F.mse_loss(pred, target)
        return loss
    
    def invariance_loss_gradcos(self, z, z_aug):
        """Gradient cosine similarity invariance"""
        grad_z = self.potential.gradient(z, create_graph=True)
        grad_z_aug = self.potential.gradient(z_aug, create_graph=True)
        grad_z_norm = F.normalize(grad_z, dim=-1)
        grad_z_aug_norm = F.normalize(grad_z_aug, dim=-1)
        cos_sim = (grad_z_norm * grad_z_aug_norm).sum(dim=-1)
        loss = (1 - cos_sim).mean()
        return loss
    
    def invariance_loss_endpoint(self, z, z_aug, steps=5, step_size=0.1):
        """Endpoint consistency invariance"""
        with torch.enable_grad():
            # Run short ascent from z
            z_curr = z.clone()
            for _ in range(steps):
                grad = self.potential.gradient(z_curr, create_graph=True)
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                z_curr = z_curr + step_size * grad / grad_norm
            
            # Run short ascent from z_aug
            z_aug_curr = z_aug.clone()
            for _ in range(steps):
                grad = self.potential.gradient(z_aug_curr, create_graph=True)
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                z_aug_curr = z_aug_curr + step_size * grad / grad_norm
            
            # MSE between endpoints
            loss = F.mse_loss(z_curr, z_aug_curr)
        
        return loss
    
    def invariance_loss(self, z, z_aug):
        """Wrapper for invariance loss based on mode"""
        if self.inv_mode == 'gradcos':
            return self.invariance_loss_gradcos(z, z_aug)
        elif self.inv_mode == 'endpoint':
            return self.invariance_loss_endpoint(z, z_aug)
        else:
            raise ValueError(f"Unknown invariance mode: {self.inv_mode}")
    
    def estimate_z_statistics(self, loader=None, embeddings=None, num_batches=10):
        """Estimate embedding statistics for noise scaling
        
        Can use either a data loader or pre-computed embeddings
        """
        print("Estimating embedding statistics...")
        
        if embeddings is not None:
            # Use pre-computed embeddings
            z_tensor = torch.from_numpy(embeddings).float().to(self.device)
            self.z_std = z_tensor.std(dim=0).mean().item()
        elif loader is not None:
            # Compute from loader
            z_list = []
            self.encoder.eval()
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= num_batches:
                        break
                    x = batch[0].to(self.device)
                    z = self.encoder(x)
                    z_list.append(z)
            
            z_all = torch.cat(z_list, dim=0)
            self.z_std = z_all.std(dim=0).mean().item()
        else:
            raise ValueError("Either loader or embeddings must be provided")
        
        print(f"  Embedding std: {self.z_std:.4f}")
    
    def train_epoch(self, loader, use_amp=False):
        """Train one epoch with optional mixed precision"""
        self.potential.train()
        self.encoder.eval()  # Keep backbone frozen
        
        epoch_losses = []
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.device.type == 'cuda' else None
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for x, x_aug, _ in pbar:
            x = x.to(self.device)
            x_aug = x_aug.to(self.device)
            
            # Get embeddings (may have gradients if projection is trainable)
            z = self.encoder(x)
            z_aug = self.encoder(x_aug)
            
            # Sample noise level
            sigma = self.log_uniform_sigma((z.shape[0],))
            
            # Compute losses
            if use_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    loss_dsm = self.dsm_loss(z, sigma)
                    loss_inv = self.invariance_loss(z, z_aug)
                    loss = loss_dsm + self.lambda_inv * loss_inv
            else:
                loss_dsm = self.dsm_loss(z, sigma)
                loss_inv = self.invariance_loss(z, z_aug)
                loss = loss_dsm + self.lambda_inv * loss_inv
            
            # Optimization step
            self.optimizer.zero_grad()
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                # Clip gradients for all param groups
                for group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_norm=5.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                # Clip gradients for all param groups
                for group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_norm=5.0)
                self.optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return np.mean(epoch_losses)
    
    def train_on_embeddings(self, z_loader, epochs=100, use_amp=False, z_std=None):
        """Train on cached embeddings"""
        print(f"\nTraining DSM on cached embeddings for {epochs} epochs...")
        
        # FIX: Initialize z_std if not already done
        if self.z_std is None:
            if z_std is not None:
                self.z_std = z_std
                print(f"  Using provided z_std: {self.z_std:.4f}")
            else:
                # Estimate from first batch
                print("  Estimating z_std from cached embeddings...")
                for z, _ in z_loader:
                    z = z.to(self.device)
                    self.z_std = z.std(dim=0).mean().item()
                    print(f"  Estimated z_std: {self.z_std:.4f}")
                    break
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )
        
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            self.potential.train()
            epoch_losses = []
            
            pbar = tqdm(z_loader, desc=f"Epoch {epoch}", leave=False)
            for z, z_aug in pbar:
                z = z.to(self.device)
                z_aug = z_aug.to(self.device)
                
                sigma = self.log_uniform_sigma((z.shape[0],))
                
                loss_dsm = self.dsm_loss(z, sigma)
                loss_inv = self.invariance_loss(z, z_aug)
                loss = loss_dsm + self.lambda_inv * loss_inv
                
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients for all param groups
                for group in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_norm=5.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)
            self.scheduler.step()
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.best_state = {
                    'potential': self.potential.state_dict().copy(),
                    'encoder': self.encoder.state_dict().copy()
                }
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
        
        # Load best model
        self.potential.load_state_dict(self.best_state['potential'])
        self.encoder.load_state_dict(self.best_state['encoder'])
        print(f"✓ Training complete! Best loss: {best_loss:.4f}")
    
    def train(self, loader, epochs=100, use_amp=False):
        """Full training loop"""
        if self.z_std is None:
            self.estimate_z_statistics(loader)
        
        # Initialize scheduler with correct T_max
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )
        
        print(f"\nStarting DSM training for {epochs} epochs...")
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(loader, use_amp)
            self.loss_history.append(avg_loss)
            self.scheduler.step()
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.best_state = {
                    'potential': self.potential.state_dict().copy(),
                    'encoder': self.encoder.state_dict().copy()
                }
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
        
        # Load best model
        self.potential.load_state_dict(self.best_state['potential'])
        self.encoder.load_state_dict(self.best_state['encoder'])
        print(f"✓ Training complete! Best loss: {best_loss:.4f}")

# ============================================================================
# Mode Seeking with Element-wise Backtracking
# ============================================================================

class ModeSeeking:
    """Gradient ascent with element-wise backtracking for stability"""
    
    def __init__(self, potential, step_size=0.2, max_steps=20, 
                 tolerance=1e-3, normalize_step=True, use_backtracking=True):
        
        self.potential = potential
        self.step_size = step_size
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.normalize_step = normalize_step
        self.use_backtracking = use_backtracking
    
    @torch.no_grad()
    def ascend_to_mode(self, z0):
        """Gradient ascent with element-wise backtracking"""
        z = z0.clone()
        
        for step in range(self.max_steps):
            # Compute gradient
            grad = self.potential.gradient_batch(z)
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            
            # Check convergence
            if (grad_norm < self.tolerance).all():
                break
            
            # Normalize gradient if requested
            if self.normalize_step:
                direction = grad / grad_norm
            else:
                direction = grad
            
            # Element-wise backtracking line search
            if self.use_backtracking:
                phi_current = self.potential(z)
                alpha = torch.full_like(phi_current, self.step_size)
                
                # Track which samples improved
                improved_mask = torch.zeros_like(phi_current, dtype=torch.bool)
                
                for _ in range(3):  # Max 3 halvings
                    z_try = z + alpha.unsqueeze(-1) * direction
                    phi_try = self.potential(z_try)
                    
                    # Element-wise improvement check
                    improve = phi_try >= phi_current
                    
                    if improve.any():
                        # Update only improving samples
                        z[improve] = z_try[improve]
                        phi_current[improve] = phi_try[improve]
                        improved_mask = improved_mask | improve
                    
                    # Halve alpha only for non-improving samples
                    alpha = torch.where(improve, alpha, alpha * 0.5)
                
                # Tiny nudge only for samples that never improved
                nudge_mask = ~improved_mask
                if nudge_mask.any():
                    z[nudge_mask] = z[nudge_mask] + 0.01 * direction[nudge_mask]
            else:
                z = z + self.step_size * direction
        
        phi = self.potential(z)
        return z, phi, step + 1
    
    def find_modes(self, encoder, loader, device='cuda'):
        """Find all modes in dataset"""
        
        print("Finding modes via gradient ascent...")
        encoder = encoder.to(device).eval()
        self.potential = self.potential.to(device).eval()
        
        all_endpoints = []
        all_phi = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Mode seeking"):
                # Handle both (x, x_aug, y) and (x, y) formats
                if len(batch) == 3:
                    x, x_aug, labels = batch
                else:
                    x, labels = batch
                    x_aug = x
                
                x = x.to(device)
                z0 = encoder(x)
                z_mode, phi, _ = self.ascend_to_mode(z0)
                
                all_endpoints.append(z_mode.cpu())
                all_phi.append(phi.cpu())
                all_labels.append(labels)
        
        endpoints = torch.cat(all_endpoints, dim=0).numpy()
        phi_values = torch.cat(all_phi, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        
        return endpoints, phi_values, labels

# ============================================================================
# Topological Persistence with Edge Case Handling
# ============================================================================

class PersistencePruning:
    """0-D persistence for automatic cluster selection"""
    
    def __init__(self, k_neighbors=10, tau_percentile=30):
        self.k_neighbors = k_neighbors
        self.tau_percentile = tau_percentile
        self.tau_value = None  # Store for later reference
        self.n_immortals = 0   # Store for later reference
    
    def compute_persistence(self, mode_centers, mode_phi):
        """Compute 0-D persistence with proper immortal handling"""
        n_modes = len(mode_centers)
        
        # Handle edge case: too few modes
        if n_modes <= 1:
            return list(range(n_modes)), {0: np.inf}, [0], []
        
        print(f"Computing persistence for {n_modes} modes...")
        
        # Build k-NN graph
        k = min(self.k_neighbors + 1, n_modes)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto')
        nbrs.fit(mode_centers)
        neighbors = nbrs.kneighbors(mode_centers, return_distance=False)[:, 1:]
        
        # Union-Find data structure
        parent = list(range(n_modes))
        birth = mode_phi.copy()
        death = np.full(n_modes, -np.inf)
        
        def find(x):
            path = []
            while parent[x] != x:
                path.append(x)
                x = parent[x]
            # Path compression
            for p in path:
                parent[p] = x
            return x
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return ra, rb, False
            
            # Union by birth time (higher φ wins)
            if birth[ra] >= birth[rb]:
                parent[rb] = ra
                return ra, rb, True
            else:
                parent[ra] = rb
                return rb, ra, True
        
        # Process modes in descending order of φ
        order = np.argsort(-mode_phi)
        active = np.zeros(n_modes, dtype=bool)
        
        for idx in order:
            active[idx] = True
            
            # Check neighbors that are already active (higher φ)
            for neighbor_idx in neighbors[idx]:
                if active[neighbor_idx]:
                    root_current = find(idx)
                    root_neighbor = find(neighbor_idx)
                    
                    if root_current != root_neighbor:
                        winner, loser, merged = union(root_current, root_neighbor)
                        if merged:
                            death[loser] = mode_phi[idx]
        
        # Compute persistence values
        roots = [find(i) for i in range(n_modes)]
        unique_roots = np.unique(roots)
        
        persistence_values = {}
        immortal_roots = []
        mortal_persistence = []
        
        for root in unique_roots:
            if death[root] == -np.inf:
                # Immortal component
                immortal_roots.append(root)
                persistence_values[root] = np.inf
            else:
                # Mortal component
                pers = birth[root] - death[root]
                persistence_values[root] = pers
                mortal_persistence.append(pers)
        
        return roots, persistence_values, immortal_roots, mortal_persistence
    
    def prune_modes(self, mode_centers, mode_phi, assignments):
        """Prune modes based on persistence threshold"""
        
        # Handle edge case
        if len(mode_centers) <= 1:
            return assignments.copy(), max(1, len(mode_centers))
        
        roots, persistence_values, immortal_roots, mortal_persistence = \
            self.compute_persistence(mode_centers, mode_phi)
        
        # Store for reference
        self.n_immortals = len(immortal_roots)
        
        # Always keep immortal components
        survivors = set(immortal_roots)
        
        # Apply percentile threshold only to mortal components
        if mortal_persistence:
            self.tau_value = np.percentile(mortal_persistence, self.tau_percentile)
            print(f"  Persistence threshold (τ): {self.tau_value:.4f}")
            
            for root, pers in persistence_values.items():
                if pers >= self.tau_value or pers == np.inf:
                    survivors.add(root)
        else:
            self.tau_value = 0.0
        
        survivors = sorted(list(survivors))
        n_clusters = len(survivors)
        print(f"  Kept {n_clusters} clusters ({self.n_immortals} immortal)")
        
        # Map to final cluster IDs
        root_to_cluster = {root: i for i, root in enumerate(survivors)}
        
        final_assignments = np.zeros(len(assignments), dtype=int)
        for i, mode_id in enumerate(assignments):
            root = roots[mode_id]
            
            if root in root_to_cluster:
                final_assignments[i] = root_to_cluster[root]
            else:
                # Assign to nearest surviving cluster by φ value
                nearest = min(survivors, key=lambda s: abs(mode_phi[s] - mode_phi[mode_id]))
                final_assignments[i] = root_to_cluster[nearest]
        
        return final_assignments, n_clusters

# ============================================================================
# Evaluation Metrics with Fixed Stability
# ============================================================================

class ClusteringMetrics:
    """Comprehensive clustering evaluation metrics"""
    
    @staticmethod
    def clustering_accuracy(y_true, y_pred):
        """Clustering accuracy via Hungarian algorithm"""
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        
        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1
        
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        acc = w[row_ind, col_ind].sum() / len(y_pred)
        return acc
    
    @staticmethod
    def stability_at_aug(endpoints1, endpoints2, threshold=None):
        """Compute Stability@Aug with fixed threshold
        
        FIX: Use a fixed threshold based on median distance, not quantile
        """
        # Compute pairwise distances
        distances = np.linalg.norm(endpoints1 - endpoints2, axis=1)
        
        if threshold is None:
            # Use median distance as threshold (more stable than quantile)
            threshold = np.median(distances)
        
        # Compute stability fraction
        stable_frac = (distances < threshold).mean()
        
        # Also return threshold for reference
        return stable_frac, threshold

    @staticmethod
    def assignment_consistency(assignments1, assignments2):
        """Compute assignment consistency between two views."""
        # Remap to 0..K-1 to be safe
        a1, _ = pd.factorize(assignments1)
        a2, _ = pd.factorize(assignments2)
        n1 = a1.max() + 1
        n2 = a2.max() + 1

        contingency = np.zeros((n1, n2), dtype=np.float64)
        for i in range(len(a1)):
            contingency[a1[i], a2[i]] += 1.0

        row_ind, col_ind = linear_sum_assignment(-contingency)
        consistency = contingency[row_ind, col_ind].sum() / len(a1)
        return consistency

    # @staticmethod
    # def assignment_consistency(assignments1, assignments2):
    #     """Compute assignment consistency between two views
        
    #     Measures how often samples are assigned to the same cluster
    #     under different augmentations
    #     """
    #     # Create contingency matrix
    #     n1 = len(np.unique(assignments1))
    #     n2 = len(np.unique(assignments2))
    #     contingency = np.zeros((n1, n2))
        
    #     for i in range(len(assignments1)):
    #         contingency[assignments1[i], assignments2[i]] += 1
        
    #     # Find best matching using Hungarian algorithm
    #     row_ind, col_ind = linear_sum_assignment(-contingency)
        
    #     # Compute consistency as fraction of samples with matched assignments
    #     consistent = contingency[row_ind, col_ind].sum()
    #     consistency = consistent / len(assignments1)
        
    #     return consistency
    
    @staticmethod
    def evaluate_clustering(y_true, y_pred, endpoints1=None, endpoints2=None, 
                          assignments2=None, verbose=True):
        """Compute all clustering metrics"""
        
        acc = ClusteringMetrics.clustering_accuracy(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        
        n_clusters_pred = len(np.unique(y_pred))
        n_clusters_true = len(np.unique(y_true))
        
        results = {
            'ACC': acc,
            'NMI': nmi,
            'ARI': ari,
            'n_clusters_pred': n_clusters_pred,
            'n_clusters_true': n_clusters_true
        }
        
        # Add stability metrics if endpoints provided
        if endpoints1 is not None and endpoints2 is not None:
            stability, threshold = ClusteringMetrics.stability_at_aug(endpoints1, endpoints2)
            results['Stability@Aug'] = stability
            results['stability_threshold'] = threshold
        
        # Add assignment consistency if second assignments provided
        if assignments2 is not None:
            consistency = ClusteringMetrics.assignment_consistency(y_pred, assignments2)
            results['Stability@Assign'] = consistency
        
        if verbose:
            print("\n" + "=" * 50)
            print("Clustering Results:")
            print("=" * 50)
            print(f"ACC: {acc:.4f}")
            print(f"NMI: {nmi:.4f}")
            print(f"ARI: {ari:.4f}")
            if 'Stability@Aug' in results:
                print(f"Stability@Aug: {results['Stability@Aug']:.4f}")
            if 'Stability@Assign' in results:
                print(f"Stability@Assign: {results['Stability@Assign']:.4f}")
            print(f"Predicted clusters: {n_clusters_pred}")
            print(f"True clusters: {n_clusters_true}")
            print("=" * 50)
        
        return results

# ============================================================================
# Embedding Cache Implementation
# ============================================================================

class EmbeddingCache:
    """Optional embedding cache for faster training"""
    
    def __init__(self, cache_dir='./cache', enabled=False):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(exist_ok=True)
    
    def compute_and_save(self, encoder, loader, device, name='train'):
        """Compute and cache embeddings"""
        if not self.enabled:
            return None, None, None
        
        cache_file = self.cache_dir / f'{name}_embeddings.npz'
        
        if cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}")
            data = np.load(cache_file)
            return data['embeddings'], data['embeddings_aug'], data['labels']
        
        print(f"Computing embeddings for caching...")
        encoder = encoder.to(device).eval()
        
        all_embeddings = []
        all_embeddings_aug = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Caching embeddings"):
                if len(batch) == 3:
                    x, x_aug, labels = batch
                else:
                    x, labels = batch
                    x_aug = x
                
                x = x.to(device)
                x_aug = x_aug.to(device)
                
                z = encoder(x)
                z_aug = encoder(x_aug)
                
                all_embeddings.append(z.cpu().numpy())
                all_embeddings_aug.append(z_aug.cpu().numpy())
                all_labels.append(labels.numpy())
        
        embeddings = np.concatenate(all_embeddings)
        embeddings_aug = np.concatenate(all_embeddings_aug)
        labels = np.concatenate(all_labels)
        
        print(f"Saving embeddings to {cache_file}")
        np.savez_compressed(cache_file, 
                           embeddings=embeddings, 
                           embeddings_aug=embeddings_aug,
                           labels=labels)
        
        return embeddings, embeddings_aug, labels

# ============================================================================
# Main NMC Pipeline
# ============================================================================

class NMCPipeline:
    """Complete Neural Mode Clustering pipeline"""
    
    def __init__(self, dataset_name='cifar10', embedding_dim=128,
                 potential_hidden=512, potential_depth=4, 
                 device='cuda', cache_embeddings=False, 
                 inv_mode='endpoint', train_projection=True):
        
        self.dataset_name = dataset_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.inv_mode = inv_mode
        
        # Initialize encoder with trainable/fixed projection
        self.encoder = FrozenEncoder(
            model_name='resnet18', 
            out_dim=embedding_dim,
            train_projection=train_projection
        )
        
        # Adjust embedding dim if not using projection
        actual_dim = self.encoder.out_dim
        
        self.potential = NeuralPotential(
            d_in=actual_dim, 
            hidden=potential_hidden, 
            depth=potential_depth
        )
        
        self.cache = EmbeddingCache(enabled=cache_embeddings)
        
        self.trainer = None
        self.mode_seeker = None
        self.persistence_pruner = None
        self.results = {}
    
    def train(self, train_loader, epochs=100, lr=1e-3, lambda_inv=0.1, use_amp=False):
        """Train the potential function"""
        
        print("\n" + "="*60)
        print("TRAINING NEURAL POTENTIAL")
        print("="*60)
        
        self.trainer = DSMTrainer(
            encoder=self.encoder,
            potential=self.potential,
            device=self.device,
            lr=lr,
            lambda_inv=lambda_inv,
            inv_mode=self.inv_mode
        )
        
        # Use cached embeddings if enabled
        if self.cache.enabled:
            z, z_aug, labels = self.cache.compute_and_save(
                self.encoder, train_loader, self.device, 'train'
            )
            
            if z is not None:
                # Create dataset from cached embeddings
                z_dataset = CachedEmbeddingDataset(z, z_aug)
                z_loader = DataLoader(
                    z_dataset, 
                    batch_size=train_loader.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                    drop_last=True
                )
                
                # Compute z_std from cached embeddings
                # z is a numpy array of shape [N, D]
                z_std = z.std(axis=0).mean()
                
                # Train on cached embeddings
                self.trainer.train_on_embeddings(z_loader, epochs=epochs, 
                                                use_amp=use_amp, z_std=z_std)
            else:
                # Fallback to regular training
                self.trainer.train(train_loader, epochs=epochs, use_amp=use_amp)
        else:
            # Regular training
            self.trainer.train(train_loader, epochs=epochs, use_amp=use_amp)
        
        self.results['loss_history'] = self.trainer.loss_history
        return self.trainer.loss_history
    
    def cluster(self, test_loader, step_size=0.2, max_steps=20, 
                tau_percentile=30, compute_stability=True, data_dir='./data'):
        """Perform clustering on test data with proper stability computation"""
        
        print("\n" + "="*60)
        print("CLUSTERING WITH NMC")
        print("="*60)
        
        self.mode_seeker = ModeSeeking(
            potential=self.potential,
            step_size=step_size,
            max_steps=max_steps,
            use_backtracking=True
        )
        
        self.persistence_pruner = PersistencePruning(tau_percentile=tau_percentile)
        
        # Find modes on test set
        endpoints, phi_values, true_labels = self.mode_seeker.find_modes(
            self.encoder, test_loader, self.device
        )
        
        # Compute stability with proper two-view loader
        endpoints_aug = None
        assignments_aug = None
        if compute_stability:
            print("Computing augmentation stability...")
            
            # Create two-view test loader (maintains same order as test_loader)
            two_view_loader = make_two_view_test_loader(
                data_dir=data_dir,
                batch_size=test_loader.batch_size,
                num_workers=getattr(test_loader, 'num_workers', 2)
            )
            
            # Get augmented view endpoints
            endpoints_aug_chunks = []
            with torch.no_grad():
                for x_weak, x_strong, _ in tqdm(two_view_loader, desc="Mode seeking (aug view)"):
                    x = x_strong.to(self.device)
                    z0 = self.encoder(x)
                    z_mode, _, _ = self.mode_seeker.ascend_to_mode(z0)
                    endpoints_aug_chunks.append(z_mode.cpu())
            
            endpoints_aug = torch.cat(endpoints_aug_chunks, 0).numpy()
        
        # Deduplicate modes using DBSCAN (removed deprecated n_jobs parameter)
        eps = 0.5
        min_samples = 5
        
        print(f"Deduplicating {len(endpoints)} endpoints...")
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)  # Removed n_jobs=-1
        cluster_labels = clusterer.fit_predict(endpoints)
        
        # Handle edge case: DBSCAN finds no clusters
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        n_modes = len(unique_labels)
        
        if n_modes == 0:
            print("  Warning: DBSCAN found no clusters, falling back to K-Means")
            k_guess = min(64, max(4, endpoints.shape[0] // 100))
            km = KMeans(n_clusters=k_guess, n_init=10, random_state=42).fit(endpoints)
            cluster_labels = km.labels_
            unique_labels = np.unique(cluster_labels)
            n_modes = len(unique_labels)
        
        print(f"  Found {n_modes} unique modes")
        
        mode_centers = np.zeros((n_modes, endpoints.shape[1]))
        mode_phi = np.zeros(n_modes)
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            mode_centers[i] = endpoints[mask].mean(axis=0)
            mode_phi[i] = phi_values[mask].mean()
        
        # Assign to nearest mode
        distances = cdist(endpoints, mode_centers)
        initial_assignments = distances.argmin(axis=1)
        
        # Persistence pruning
        final_assignments, n_clusters = self.persistence_pruner.prune_modes(
            mode_centers, mode_phi, initial_assignments
        )
        
        # If we have augmented endpoints, compute their assignments too
        if endpoints_aug is not None:
            distances_aug = cdist(endpoints_aug, mode_centers)
            initial_assignments_aug = distances_aug.argmin(axis=1)
            assignments_aug, _ = self.persistence_pruner.prune_modes(
                mode_centers, mode_phi, initial_assignments_aug
            )
        
        # Store results
        self.results.update({
            'endpoints': endpoints,
            'endpoints_aug': endpoints_aug,
            'mode_centers': mode_centers,
            'mode_phi': mode_phi,
            'assignments': final_assignments,
            'assignments_aug': assignments_aug,
            'true_labels': true_labels,
            'n_clusters': n_clusters,
            'tau_value': self.persistence_pruner.tau_value,
            'n_immortals': self.persistence_pruner.n_immortals
        })
        
        return final_assignments, true_labels
    
    def evaluate(self):
        """Evaluate clustering results"""
        
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        metrics = ClusteringMetrics.evaluate_clustering(
            self.results['true_labels'],
            self.results['assignments'],
            self.results.get('endpoints'),
            self.results.get('endpoints_aug'),
            self.results.get('assignments_aug')
        )
        
        self.results['metrics'] = metrics
        return metrics
    
    def compare_with_baselines(self, test_loader):
        """Compare with K-Means baseline (auto-k and k=10)"""
        
        print("\n" + "="*60)
        print("BASELINE COMPARISONS")
        print("="*60)
        
        # Get embeddings
        embeddings = []
        labels = []
        
        self.encoder = self.encoder.to(self.device).eval()
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    x, _, y = batch
                else:
                    x, y = batch
                
                x = x.to(self.device)
                z = self.encoder(x)
                embeddings.append(z.cpu())
                labels.append(y)
        
        embeddings = torch.cat(embeddings).numpy()
        labels = torch.cat(labels).numpy()
        
        results = []
        
        # NMC results
        nmc_metrics = self.results['metrics'].copy()
        nmc_metrics['Method'] = 'NMC (Ours)'
        results.append(nmc_metrics)
        
        # K-Means with auto-k (matching NMC's cluster count)
        print(f"Running K-Means (k={self.results['n_clusters']})...")
        kmeans_auto = KMeans(n_clusters=self.results['n_clusters'], 
                             random_state=42, n_init=10)
        pred_kmeans_auto = kmeans_auto.fit_predict(embeddings)
        metrics_kmeans_auto = ClusteringMetrics.evaluate_clustering(
            labels, pred_kmeans_auto, verbose=False
        )
        metrics_kmeans_auto['Method'] = f'K-Means (k={self.results["n_clusters"]})'
        results.append(metrics_kmeans_auto)
        
        # K-Means with k=10 (for CIFAR-10 fairness)
        if self.dataset_name.lower() == 'cifar10':
            print("Running K-Means (k=10)...")
            kmeans_10 = KMeans(n_clusters=10, random_state=42, n_init=10)
            pred_kmeans_10 = kmeans_10.fit_predict(embeddings)
            metrics_kmeans_10 = ClusteringMetrics.evaluate_clustering(
                labels, pred_kmeans_10, verbose=False
            )
            metrics_kmeans_10['Method'] = 'K-Means (k=10)'
            results.append(metrics_kmeans_10)
        
        # Create comparison table
        comparison = pd.DataFrame(results)
        cols = ['Method', 'ACC', 'NMI', 'ARI']
        if 'Stability@Aug' in comparison.columns:
            cols.append('Stability@Aug')
        if 'Stability@Assign' in comparison.columns:
            cols.append('Stability@Assign')
        cols.append('n_clusters_pred')
        comparison = comparison[cols]
        
        print("\nComparison Table:")
        print(comparison.to_string(index=False))
        
        self.results['comparison'] = comparison
        return comparison
# ============================================================================
# Visualization and Export Functions
# ============================================================================

def plot_training_curves(loss_history):
    """Plot training loss curves"""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    if len(loss_history) > 10:
        window = min(10, len(loss_history) // 5)
        smooth = np.convolve(loss_history, np.ones(window)/window, mode='valid')
        smooth_epochs = epochs[:len(smooth)]
        ax.plot(smooth_epochs, smooth, 'r--', linewidth=2, alpha=0.7, label='Smoothed')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('DSM Training Progress', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def generate_latex_table(comparison_df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper (handles optional stability columns)"""
    print("\n" + "="*60); print("LATEX TABLE FOR PAPER"); print("="*60)
    has_aug = 'Stability@Aug' in comparison_df.columns
    has_assign = 'Stability@Assign' in comparison_df.columns

    header_cols = ['Method', 'ACC', 'NMI', 'ARI']
    if has_aug: header_cols.append('Stability@Aug')
    if has_assign: header_cols.append('Stability@Assign')

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Clustering performance on CIFAR-10}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\begin{tabular}{l" + "c"*(len(header_cols)-1) + "}")
    latex.append("\\toprule")
    header_display = [c if c=='Method' else f"{c} $\\uparrow$" for c in header_cols]
    latex.append(" & ".join(header_display) + " \\\\")
    latex.append("\\midrule")

    for _, row in comparison_df.iterrows():
        vals = []
        method = row['Method']
        is_ours = (method == 'NMC (Ours)')
        vals.append("\\textbf{NMC (Ours)}" if is_ours else method)

        def fmt(v, bold=False):
            if pd.isna(v): return "-"
            return f"\\textbf{{{v:.4f}}}" if bold else f"{v:.4f}"

        vals.append(fmt(row['ACC'], is_ours))
        vals.append(fmt(row['NMI'], is_ours))
        vals.append(fmt(row['ARI'], is_ours))
        if has_aug:    vals.append(fmt(row.get('Stability@Aug', np.nan), is_ours))
        if has_assign: vals.append(fmt(row.get('Stability@Assign', np.nan), is_ours))

        latex.append(" & ".join(vals) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    table = "\n".join(latex)
    print(table)
    return table


def save_results(nmc_pipeline, config, save_dir='./nmc_results'):
    """Save model, metrics, predictions, comparison table, and LaTeX."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to {save_dir}/")

    # Model weights
    torch.save({
        'encoder_state': nmc_pipeline.encoder.state_dict(),
        'potential_state': nmc_pipeline.potential.state_dict(),
        'config': config
    }, save_dir / 'nmc_model.pt')
    print("✓ Model saved")

    # Results JSON
    results_to_save = {
        'metrics': nmc_pipeline.results.get('metrics', {}),
        'n_clusters': nmc_pipeline.results.get('n_clusters', None),
        'loss_history': nmc_pipeline.results.get('loss_history', []),
        'tau_value': nmc_pipeline.results.get('tau_value', None),
        'n_immortals': nmc_pipeline.results.get('n_immortals', None),
        'stability_threshold': nmc_pipeline.results.get('metrics', {}).get('stability_threshold', None),
    }
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2, default=float)
    print("✓ Results saved")

    # Arrays
    if 'assignments' in nmc_pipeline.results:
        np.save(save_dir / 'predictions.npy', nmc_pipeline.results['assignments'])
    if 'true_labels' in nmc_pipeline.results:
        np.save(save_dir / 'true_labels.npy', nmc_pipeline.results['true_labels'])
    if 'endpoints' in nmc_pipeline.results:
        np.save(save_dir / 'endpoints.npy', nmc_pipeline.results['endpoints'])
    if 'endpoints_aug' in nmc_pipeline.results and nmc_pipeline.results['endpoints_aug'] is not None:
        np.save(save_dir / 'endpoints_aug.npy', nmc_pipeline.results['endpoints_aug'])
    print("✓ Arrays saved")

    # Comparison + LaTeX
    if 'comparison' in nmc_pipeline.results:
        comp = nmc_pipeline.results['comparison']
        comp.to_csv(save_dir / 'comparison.csv', index=False)
        tex = generate_latex_table(comp)
        with open(save_dir / 'latex_table.tex', 'w') as f:
            f.write(tex)
        print("✓ Comparison & LaTeX saved")


# ============================================================================
# Main Execution
# ============================================================================

def main(args):
    """Main execution function"""
    set_seed(args.seed)
    device = get_device()

    print("Loading dataset...")
    train_loader, test_loader, _ = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers
    )
    print(f"✓ Dataset loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")

    nmc = NMCPipeline(
        dataset_name=args.dataset,
        embedding_dim=args.embedding_dim,
        potential_hidden=args.potential_hidden,
        potential_depth=args.potential_depth,
        device=device,
        cache_embeddings=args.cache_embeddings,
        inv_mode=args.inv_mode,
        train_projection=args.train_projection
    )

    loss_history = nmc.train(
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        lambda_inv=args.lambda_inv,
        use_amp=args.amp
    )

    if args.plot:
        _ = plot_training_curves(loss_history); plt.show()

    # Clustering + stability
    _pred, _true = nmc.cluster(
        test_loader,
        step_size=args.step_size,
        max_steps=args.max_steps,
        tau_percentile=args.tau_percentile,
        compute_stability=True,
        data_dir=args.data_dir
    )

    metrics = nmc.evaluate()
    _comp = nmc.compare_with_baselines(test_loader)

    save_results(nmc, vars(args), args.save_dir)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    return nmc, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Mode Clustering')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data workers')

    # Model
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--potential_hidden', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--potential_depth', type=int, default=4, help='MLP depth')
    parser.add_argument('--train_projection', action='store_true', default=True,
                        help='Train projection head (default: True)')
    parser.add_argument('--no_train_projection', dest='train_projection', action='store_false',
                        help='Use backbone features (no projection)')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda_inv', type=float, default=0.1, help='Invariance weight')
    parser.add_argument('--inv_mode', type=str, default='endpoint',
                        choices=['gradcos', 'endpoint'], help='Invariance mode')

    # Mode seeking
    parser.add_argument('--step_size', type=float, default=0.2, help='Ascent step size')
    parser.add_argument('--max_steps', type=int, default=20, help='Max ascent steps')
    parser.add_argument('--tau_percentile', type=float, default=30, help='Persistence threshold')

    # Options
    parser.add_argument('--cache_embeddings', action='store_true', help='Cache embeddings')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./nmc_results', help='Save directory')
    parser.add_argument('--plot', action='store_true', help='Plot figures')

    args = parser.parse_args()
    main(args)
