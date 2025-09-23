# SM64 AI Agent System

A Vision Transformer-based agent that learns to play Super Mario 64 through behavior cloning from demonstration data.

## Architecture

- **Vision Model**: ViT-B/16 pretrained on ImageNet, fine-tuned for SM64
- **Action Space**:
  - 16 discrete buttons (multi-label classification)
  - 2D continuous stick position (regression)
- **Real-time Inference**: 15 FPS with action interpolation
- **WebSocket Integration**: Direct game control through network protocol

## Setup

```bash
pip install -r requirements.txt
```

## Training

Train the agent on collected demonstration data:

```bash
python train.py --data-dir data_store/ --epochs 100 --batch-size 32
```

Key arguments:
- `--data-dir`: Directory containing CSV files and framebuffers
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--device`: cuda/cpu
- `--checkpoint`: Resume from checkpoint
- `--frame-skip`: Process every Nth frame (default: 2)

## Deployment

Run the trained agent in real-time:

```bash
python play.py --model checkpoints/best_model.pt --server ws://localhost:8080
```

Key arguments:
- `--model`: Path to trained model checkpoint
- `--server`: WebSocket server URI
- `--debug`: Enable debug mode with action logging
- `--fps`: Target inference FPS (default: 15)

## Data Format

The agent expects data in the following structure:
```
data_store/
├── *.csv (action labels)
└── framebuffers/
    └── [session_id]/
        └── *.bin (raw RGB framebuffers)
```

CSV format:
```csv
Buttons,Stick_X,Stick_Y,Framebuffer_Path
0000,0,0,data_store/framebuffers/session/1.bin
```

## Components

- `model.py`: ViT-based policy network with dual-head output
- `data_loader.py`: Dataset for loading framebuffers and actions
- `trainer.py`: Training loop with behavior cloning
- `inference_engine.py`: Real-time inference with action buffering
- `websocket_agent.py`: WebSocket client for game interaction
- `utils.py`: Evaluation and checkpoint management
- `train.py`: Main training script
- `play.py`: Deployment script for real-time play

## Performance Optimizations

- **Frame Skipping**: Process every 2nd frame to reduce compute
- **Action Interpolation**: Smooth transitions between predictions
- **Temporal Smoothing**: EMA filtering for stick movements
- **Action Buffering**: Handle inference latency with predictive buffering

## Notes

- The model uses behavior cloning, so quality depends on demonstration data
- Requires CUDA GPU for real-time inference at 15 FPS
- WebSocket server must be running before starting the agent
- Action smoothing helps with jittery controls but adds ~33ms latency