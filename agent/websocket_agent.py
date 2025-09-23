import asyncio
import websockets
import struct
import numpy as np
import torch
from pathlib import Path
import time
import signal
import sys
from inference_engine import RealtimeInferenceEngine, FrameInterpolator
from model import SM64Agent


class SM64WebSocketAgent:
    def __init__(self, model_path, server_uri="ws://localhost:8080", device='cuda'):
        self.server_uri = server_uri
        self.device = device

        self.model = SM64Agent()
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model checkpoint found, using random weights")

        self.inference_engine = RealtimeInferenceEngine(
            self.model, device=device, inference_fps=15
        )
        self.frame_interpolator = FrameInterpolator(interpolation_frames=2)

        self.running = False
        self.websocket = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0

        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame_param):
        print("\nShutting down gracefully...")
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        sys.exit(0)

    def parse_framebuffer(self, data):
        width, height = 320, 240
        expected_size = width * height * 3

        if len(data) != expected_size:
            print(f"Framebuffer size mismatch: expected {expected_size}, got {len(data)}")
            return None

        try:
            pixels = struct.unpack(f'{width*height*3}B', data)
            image_array = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
            return image_array
        except Exception as e:
            print(f"Error parsing framebuffer: {e}")
            return None

    def format_action_message(self, action):
        if action is None:
            return "0000,0,0"

        buttons_hex = f"{action['buttons']:04X}"
        stick_x = max(-128, min(127, action['stick_x']))
        stick_y = max(-128, min(127, action['stick_y']))

        return f"{buttons_hex},{stick_x},{stick_y}"

    async def handle_frame(self, websocket):
        self.running = True
        self.inference_engine.start()

        last_status_time = time.time()

        try:
            while self.running:
                try:
                    frame_data = await asyncio.wait_for(websocket.recv(), timeout=0.1)

                    if isinstance(frame_data, bytes):
                        frame = self.parse_framebuffer(frame_data)
                        if frame is not None:
                            action = self.inference_engine.process_frame(frame)

                            if action:
                                self.frame_interpolator.set_target(action)

                            interpolated_action = self.frame_interpolator.get_current_action()
                            if interpolated_action:
                                message = self.format_action_message(interpolated_action)
                                await websocket.send(message)

                            self.frame_count += 1
                            current_time = time.time()
                            if current_time - self.last_frame_time > 0:
                                self.fps = 1.0 / (current_time - self.last_frame_time)
                            self.last_frame_time = current_time

                            if current_time - last_status_time > 1.0:
                                stats = self.inference_engine.get_stats()
                                print(f"FPS: {self.fps:.1f} | Inference: {stats['inference_time_ms']:.1f}ms | Frames: {self.frame_count}")
                                last_status_time = current_time

                except asyncio.TimeoutError:
                    await websocket.send("0000,0,0")
                except Exception as e:
                    print(f"Error in frame handling: {e}")

        finally:
            self.inference_engine.stop()

    async def connect_and_play(self):
        print(f"Connecting to {self.server_uri}...")

        try:
            async with websockets.connect(self.server_uri, subprotocols=[]) as websocket:
                self.websocket = websocket
                print("Connected! Starting agent...")

                await websocket.send("AGENT_READY")

                await self.handle_frame(websocket)

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.running = False

    def run(self):
        asyncio.run(self.connect_and_play())


class DebugAgent(SM64WebSocketAgent):
    def __init__(self, model_path, server_uri="ws://localhost:8080", device='cuda'):
        super().__init__(model_path, server_uri, device)
        self.action_log = []
        self.debug_mode = True

    def format_action_message(self, action):
        message = super().format_action_message(action)

        if self.debug_mode and action:
            button_names = self.decode_buttons(action['buttons'])
            if button_names:
                print(f"Actions: {', '.join(button_names)} | Stick: ({action['stick_x']}, {action['stick_y']})")

        self.action_log.append({
            'time': time.time(),
            'action': action
        })

        return message

    def decode_buttons(self, button_value):
        button_map = {
            0x0001: 'A',
            0x0002: 'B',
            0x0004: 'Start',
            0x0008: 'L',
            0x0010: 'R',
            0x0020: 'Z',
            0x0040: 'C-Up',
            0x0080: 'C-Down',
            0x0100: 'C-Left',
            0x0200: 'C-Right',
            0x0400: 'D-Up',
            0x0800: 'D-Down',
            0x1000: 'D-Left',
            0x2000: 'D-Right'
        }

        pressed = []
        for mask, name in button_map.items():
            if button_value & mask:
                pressed.append(name)
        return pressed