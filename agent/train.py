#!/usr/bin/env python3

import argparse
import torch
from torch.utils.data import random_split
from pathlib import Path

from model import SM64Agent
from data_loader import SM64Dataset
from trainer import SM64Trainer
from utils import ModelEvaluator, compute_dataset_statistics


def main():
    parser = argparse.ArgumentParser(description='Train SM64 Agent')
    parser.add_argument('--data-dir', type=str, default='data_store/',
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--frame-skip', type=int, default=2,
                       help='Frame skip for data loading')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    print(f"Training SM64 Agent on {args.device}")
    print(f"Data directory: {args.data_dir}")

    dataset = SM64Dataset(data_dir=args.data_dir, frame_skip=args.frame_skip)
    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: No data found in the specified directory")
        return

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    print("\nComputing dataset statistics...")
    stats = compute_dataset_statistics(train_loader)
    print(f"Average stick X: {stats['stick_x_mean']:.3f} ± {stats['stick_x_std']:.3f}")
    print(f"Average stick Y: {stats['stick_y_mean']:.3f} ± {stats['stick_y_std']:.3f}")

    most_used_buttons = []
    for i, freq in enumerate(stats['button_frequencies']):
        if freq > 0.01:
            most_used_buttons.append(f"Button_{i}: {freq:.2%}")
    if most_used_buttons:
        print(f"Most used buttons: {', '.join(most_used_buttons[:5])}")

    model = SM64Agent()
    trainer = SM64Trainer(model, device=args.device)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            start_epoch, _ = trainer.load_checkpoint(checkpoint_path)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint {args.checkpoint} not found, starting from scratch")

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 50)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_freq=args.save_freq
    )

    print("\n" + "=" * 50)
    print("Training complete!")

    print("\nEvaluating final model...")
    evaluator = ModelEvaluator(model, device=args.device)
    final_metrics = evaluator.evaluate_accuracy(val_loader)

    print(f"Final Button Accuracy: {final_metrics['button_accuracy']:.4f}")
    print(f"Final Stick Error: {final_metrics['avg_stick_error']:.4f} ± {final_metrics['stick_error_std']:.4f}")

    print(f"\nModel checkpoints saved in: checkpoints/")
    print(f"Best model saved as: checkpoints/best_model.pt")


if __name__ == "__main__":
    main()