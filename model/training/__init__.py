"""
PublicSpeak PSL Model Training Module

This module contains the training functionality for the PSL model.
It handles model training, weight learning, and weight saving.
"""

from .train import main, learn, add_data, add_predicates, add_rules, write_weights

__all__ = ['main', 'learn', 'add_data', 'add_predicates', 'add_rules', 'write_weights'] 