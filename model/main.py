#!/usr/bin/env python3
"""
PublicSpeak PSL Model Main Controller

This script serves as the main entry point for the PSL model package.
It provides a unified interface for training and inference operations.

Usage:
    python -m model.main --mode train --city AA --output output
    python -m model.main --mode infer --city AA --output output
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='PublicSpeak PSL Model')
    parser.add_argument('--mode', choices=['train', 'infer', 'paper_reproduce'], required=True,
                       help='Mode: train, infer, or paper_reproduce')
    parser.add_argument('--output', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            print(f"Starting training")
            from .training.train import main as train_main
            train_main(args)
            print(f"Training completed")
            
        elif args.mode == 'infer':
            print(f"Starting inference")
            from .inference.infer import main as infer_main
            infer_main(args)
            print(f"Inference completed")
            
        elif args.mode == 'paper_reproduce':
            print(f"Starting paper reproduction")
            from .paper_reproduce.infer import main as paper_infer_main
            paper_infer_main(args)
            print(f"Paper reproduction completed")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the publicspeak directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error during {args.mode}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 