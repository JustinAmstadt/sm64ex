#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Fix numpy compatibility issue
import numpy.core._multiarray_umath
import numpy.core.multiarray
np.ndarray = np.core.multiarray.ndarray

from model import SM64Agent
from data_loader import SM64Dataset
from trainer import SM64Trainer

def main():
    print("Training SM64 Agent (Simple Mode)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset with fewer workers to avoid multiprocessing issues
    dataset = SM64Dataset(data_dir='data_store/', frame_skip=50)
    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: No data found")
        return

    # Split dataset
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders with num_workers=0 to avoid multiprocessing issues
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # No multiprocessing
        pin_memory=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # No multiprocessing
        pin_memory=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model and trainer
    model = SM64Agent()
    trainer = SM64Trainer(model, device=device)

    # Quick test of data loading
    print("\nTesting data loading...")
    try:
        for i, batch in enumerate(train_loader):
            print(f"Batch {i}: images shape = {batch['image'].shape}")
            print(f"         buttons shape = {batch['buttons'].shape}")
            print(f"         stick shape = {batch['stick'].shape}")
            if i >= 2:  # Just test a few batches
                break
        print("Data loading successful!")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # Train for 1 epoch as a test
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        save_freq=1
    )

    print("\nTraining complete!")
    print("Model saved to: checkpoints/best_model.pt")

if __name__ == "__main__":
    main()