"""
PublicSpeak PSL Model Package

This package contains the Probabilistic Soft Logic (PSL) models for public comment classification.
It includes training and inference modules for the PublicSpeak project.

Modules:
- training: Model training functionality
- inference: Model inference and evaluation functionality
- paper_reproduce: Paper reproduction functionality
"""

__version__ = "1.0.0"
__author__ = "PublicSpeak Team"

# Import main functions for easy access
from .training.train import main as train_main
from .inference.infer import main as infer_main
from .paper_reproduce.infer import main as paper_infer_main

__all__ = ['train_main', 'infer_main', 'paper_infer_main'] 