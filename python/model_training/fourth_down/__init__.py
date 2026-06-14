"""CFB model-training Track 2: fourth-down yards-gained model (5-feat, 76-class multi:softprob).

Usage:
    from model_training.fourth_down import train_fd_model, derive_fd_features, FD_PARAMS, FD_FEATURES
    model = train_fd_model(plays_df)
    model.save_model("fd_model.ubj")
"""
from __future__ import annotations

__version__ = "0.1.0"

from .constants import FD_FEATURES, FD_NROUNDS, FD_NUM_CLASS, FD_PARAMS
from .features import derive_fd_features
from .train import train_fd_model, train_from_plays

__all__ = [
    "FD_FEATURES",
    "FD_NROUNDS",
    "FD_NUM_CLASS",
    "FD_PARAMS",
    "derive_fd_features",
    "train_fd_model",
    "train_from_plays",
]
