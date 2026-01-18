from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

SEED = 20030910
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 256
    lr: float = 3e-4
    max_epochs: int = 50
    num_workers: int = 0
    weight_decay: float = 1e-4
    patience: int = 5
    device: torch.device = field(default_factory=lambda: DEVICE)

class NeuralBetaDataset(Dataset):
    def __init__(self, X, r_next, mkt_next):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.r = torch.as_tensor(np.asarray(r_next), dtype=torch.float32).view(-1)
        self.m = torch.as_tensor(np.asarray(mkt_next), dtype=torch.float32).view(-1)
        
        if len(self.X) != len(self.r) or len(self.X) != len(self.m):
            raise ValueError("Length mismatch among X, r_next, mkt_next")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            "x": self.X[idx], 
            "r_next": self.r[idx], 
            "mkt_next": self.m[idx]
        }

class NeuralBetaMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=8, activation="relu"):
        super().__init__()
        
        acts = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "identity": nn.Identity(),
            "linear": nn.Identity(),
            "sigmoid": nn.Sigmoid(),
        }
        act = acts.get(activation, nn.ReLU())
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 1) 
        )
    
    def forward(self, x):
        beta_hat = self.net(x)
        return beta_hat.squeeze(-1)
    
class NeuralBetaLoss(nn.Module):
    def __init__(self, eps:float = 1e-12):
        super().__init__()
        self.eps = eps
        
    def forward(self, beta_hat: torch.Tensor, mkt_next: torch.Tensor, r_next: torch.Tensor) -> torch.Tensor:
        y_hat = beta_hat * mkt_next
        mse = torch.mean((r_next - y_hat) ** 2)
        return torch.sqrt(mse + self.eps)
    
class NeuralBetaTrainer:
    def __init__(self, model:nn.Module):
        self.model = model
        self.config = TrainingConfig()
        self.loss_fn = NeuralBetaLoss()
        self.device = self.config.device
        self.model.to(self.device)

        self.optimizer = None
        self.scheduler = None
        self.history = {"train": [], "val": [], "lr": []}
        self.best_val_loss = float("inf")
        self.best_state = None

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        return self.optimizer.param_groups[0]["lr"] # type: ignore
    
    def train_one_epoch(self, loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            x = batch["x"].to(self.device)
            r = batch["r_next"].to(self.device)
            m = batch["mkt_next"].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True) # type: ignore
            beta_hat = self.model(x)
            loss = self.loss_fn(beta_hat, m, r)
            loss.backward()
            self.optimizer.step() # type: ignore
            
            running_loss += loss.item()
            n_batches += 1
        
        return running_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader)->float:
        """Evaluate model on validation/test set."""
        self.model.eval()
        running_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            x = batch["x"].to(self.device)
            r = batch["r_next"].to(self.device)
            m = batch["mkt_next"].to(self.device)
            
            beta_hat = self.model(x)
            loss = self.loss_fn(beta_hat, m, r)
            
            running_loss += loss.item()
            n_batches += 1
        
        return running_loss / max(n_batches, 1)

    def fit(self, train_ds: Dataset, val_ds: Dataset):
        self.model.to(DEVICE)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers, 
            pin_memory=(self.device.type == "cuda")
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers, 
            pin_memory=(self.device.type == "cuda")
        )
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        
        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        
        for epoch in range(1, self.config.max_epochs + 1):
            tr = self.train_one_epoch(train_loader)
            va = self.evaluate(val_loader)
            self.scheduler.step(va)
            
            self.history["train"].append(tr)
            self.history["val"].append(va)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            if va < best_val - 1e-6:
                best_val = va
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.patience:
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {"model": self.model, "history": self.history, "best_val": best_val}