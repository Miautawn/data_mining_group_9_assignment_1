from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def get_dataloader(dataset_kwargs: dict, dataloader_kwargs: dict) -> DataLoader:
    dataset = RnnDataset(**dataset_kwargs)
    return DataLoader(dataset, **dataloader_kwargs)


class RnnDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
    ):
        self.X = []
        self.y = []
        self.user_ids = []
        for name, group in df.groupby("window_id"):
            self.X.append(
                group.drop(
                    columns=["id", "window_id", "date", "split", "target_mean_mood"]
                ).values
            )
            self.y.append(group[label_col].iloc[0])
            self.user_ids.append(group["id"].iloc[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            self.user_ids[idx],
        )


class RnnRegressor(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=3,
            dropout=0.2,
        )

        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SELU(),
            nn.AlphaDropout(0.3),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights for the SELU-based regressor using LeCun Normal."""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                # Using PyTorch's built-in trick for LeCun Normal
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seq_data):
        lstm_out, _ = self.lstm(seq_data)
        energy = self.attention_weights(lstm_out)
        weights = F.softmax(energy, dim=1)
        context_vector = torch.sum(lstm_out * weights, dim=1)

        combined = self.dropout(context_vector)
        return self.regressor(combined)


class RnnClassifier(nn.Module):
    def __init__(self, n_classes, n_features, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=3,
            dropout=0.2,
        )

        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SELU(),
            nn.AlphaDropout(0.3),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.AlphaDropout(0.1),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights for the SELU-based classifier using LeCun Normal."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                # Using PyTorch's built-in trick for LeCun Normal
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seq_data):
        lstm_out, _ = self.lstm(seq_data)
        energy = self.attention_weights(lstm_out)
        weights = F.softmax(energy, dim=1)
        context_vector = torch.sum(lstm_out * weights, dim=1)

        combined = self.dropout(context_vector)
        return self.classifier(combined)


class EarlyStopping:
    def __init__(self, path: Path, patience: int = 3):
        """Early stopping helper class to save the best model during training.

        Args:
            patience (int, optional): Number of epochs with no improvement
                after which training will be stopped. Defaults to 3.
            path (str, optional): Path to save the best model. Defaults to "best_model.pth".
        """
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Check if early stopping condition is met.

        Args:
            val_loss (float): current validation loss
            model (torch.nn.Module): model to save

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0

            torch.save(model, self.path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
