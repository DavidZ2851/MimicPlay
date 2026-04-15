#!/usr/bin/env python3
"""
Analyze MimicPlay training log and plot validation loss.
Usage: python analyze_log.py <log.txt> [--output plot.png]
"""

import re
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """Parse log.txt and extract validation metrics."""
    train_epochs = []
    train_log_likelihood = []
    
    val_epochs = []
    val_log_likelihood = []
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Pattern: "Train Epoch XXXX" followed by JSON block
    train_pattern = r'Train Epoch (\d+)\s*\n\{([^}]+)\}'
    val_pattern = r'Validation Epoch (\d+)\s*\n\{([^}]+)\}'
    
    # Find all train epochs
    for match in re.finditer(train_pattern, content):
        epoch = int(match.group(1))
        json_str = '{' + match.group(2) + '}'
        try:
            data = json.loads(json_str)
            train_epochs.append(epoch)
            train_log_likelihood.append(data.get('Log_Likelihood', None))
        except json.JSONDecodeError:
            continue
    
    # Find all validation epochs
    for match in re.finditer(val_pattern, content):
        epoch = int(match.group(1))
        json_str = '{' + match.group(2) + '}'
        try:
            data = json.loads(json_str)
            val_epochs.append(epoch)
            val_log_likelihood.append(data.get('Log_Likelihood', None))
        except json.JSONDecodeError:
            continue
    
    return {
        'train_epochs': train_epochs,
        'train_log_likelihood': train_log_likelihood,
        'val_epochs': val_epochs,
        'val_log_likelihood': val_log_likelihood,
    }


def plot_log_likelihood(data, output_path=None, title="MimicPlay Training Progress"):
    """Plot training and validation log-likelihood over epochs."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if data['train_epochs'] and data['train_log_likelihood']:
        ax.plot(data['train_epochs'], data['train_log_likelihood'], 'b-', label='Train', alpha=0.7, linewidth=0.5)
    
    if data['val_epochs'] and data['val_log_likelihood']:
        ax.plot(data['val_epochs'], data['val_log_likelihood'], 'r-', label='Validation', alpha=0.9, linewidth=1)
        
        # Mark max validation log-likelihood
        max_idx = np.argmax(data['val_log_likelihood'])
        max_epoch = data['val_epochs'][max_idx]
        max_ll = data['val_log_likelihood'][max_idx]
        ax.scatter([max_epoch], [max_ll], color='green', s=100, zorder=5, marker='*')
        ax.annotate(f'Max: {max_ll:.2f}\nEpoch {max_epoch}',
                    xy=(max_epoch, max_ll),
                    xytext=(max_epoch + len(data['val_epochs'])*0.05, max_ll),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    fontsize=10, color='green')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Log-Likelihood', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    return fig


def print_summary(data):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    if data['train_epochs']:
        print(f"\nTrain:")
        print(f"  Epochs: {min(data['train_epochs'])} - {max(data['train_epochs'])}")
        print(f"  Final Log-Likelihood: {data['train_log_likelihood'][-1]:.4f}")
    
    if data['val_epochs']:
        print(f"\nValidation:")
        print(f"  Epochs: {min(data['val_epochs'])} - {max(data['val_epochs'])}")
        
        max_idx = np.argmax(data['val_log_likelihood'])
        print(f"  Max Log-Likelihood: {data['val_log_likelihood'][max_idx]:.4f} (epoch {data['val_epochs'][max_idx]})")
        print(f"  Final Log-Likelihood: {data['val_log_likelihood'][-1]:.4f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze MimicPlay training log")
    parser.add_argument("log_path", help="Path to log.txt")
    parser.add_argument("--output", "-o", default=None, help="Output plot path")
    parser.add_argument("--title", "-t", default="Log-Likelihood over Epochs", help="Plot title")
    
    args = parser.parse_args()
    
    print(f"Parsing log file: {args.log_path}")
    data = parse_log(args.log_path)
    
    print(f"Found {len(data['train_epochs'])} train, {len(data['val_epochs'])} validation entries")
    
    if not data['val_epochs']:
        print("No data found!")
        return
    
    print_summary(data)
    plot_log_likelihood(data, args.output, args.title)


if __name__ == "__main__":
    main()