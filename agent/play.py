#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path

from websocket_agent import SM64WebSocketAgent, DebugAgent
from model import SM64Agent


def main():
    parser = argparse.ArgumentParser(description='Deploy SM64 Agent for real-time play')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--server', type=str, default='ws://localhost:8080',
                       help='WebSocket server URI')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with action logging')
    parser.add_argument('--fps', type=int, default=15,
                       help='Target inference FPS')

    args = parser.parse_args()

    print("=" * 50)
    print("SM64 AI Agent - Real-time Player")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Server: {args.server}")
    print(f"Device: {args.device}")
    print(f"Target FPS: {args.fps}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nWarning: Model checkpoint not found at {args.model}")
        print("The agent will use random weights (untrained model)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    print("\nInitializing agent...")

    if args.debug:
        agent = DebugAgent(
            model_path=args.model,
            server_uri=args.server,
            device=args.device
        )
        print("Debug mode enabled - actions will be logged")
    else:
        agent = SM64WebSocketAgent(
            model_path=args.model,
            server_uri=args.server,
            device=args.device
        )

    agent.inference_engine.inference_fps = args.fps

    print("\nControls:")
    print("  - The agent will automatically connect and start playing")
    print("  - Press Ctrl+C to stop the agent")
    print("\nConnecting to game server...")

    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n\nAgent stopped by user")
    except Exception as e:
        print(f"\nError: {e}")

    if args.debug and hasattr(agent, 'action_log'):
        print(f"\nTotal actions logged: {len(agent.action_log)}")
        save_log = input("Save action log? (y/n): ")
        if save_log.lower() == 'y':
            import json
            log_path = Path('agent_log.json')
            with open(log_path, 'w') as f:
                json.dump(agent.action_log, f)
            print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()