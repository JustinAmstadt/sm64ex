import torch
import numpy as np
from collections import deque
import time
from threading import Thread, Lock
from queue import Queue
import cv2
from torchvision import transforms
from PIL import Image


class RealtimeInferenceEngine:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 frame_buffer_size=3, inference_fps=15):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.action_buffer = deque(maxlen=5)
        self.inference_fps = inference_fps
        self.frame_interval = 1.0 / inference_fps

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.frame_queue = Queue(maxsize=10)
        self.action_queue = Queue(maxsize=10)

        self.running = False
        self.inference_thread = None
        self.lock = Lock()

        self.last_action = None
        self.action_repeat = 2
        self.action_counter = 0

        self.stats = {
            'frames_processed': 0,
            'inference_time_ms': 0,
            'fps': 0
        }

    def preprocess_frame(self, frame):
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            image = Image.fromarray(frame)
        else:
            image = frame

        return self.transform(image).unsqueeze(0).to(self.device)

    def process_frame(self, frame):
        if not self.running:
            return None

        try:
            self.frame_queue.put(frame, timeout=0.01)
        except:
            pass

        try:
            action = self.action_queue.get(timeout=0.01)
            return action
        except:
            return self.last_action

    def inference_loop(self):
        last_inference_time = time.time()

        while self.running:
            current_time = time.time()

            if current_time - last_inference_time < self.frame_interval:
                time.sleep(0.001)
                continue

            try:
                frame = self.frame_queue.get(timeout=0.1)
            except:
                continue

            start_time = time.time()

            preprocessed_frame = self.preprocess_frame(frame)

            with torch.no_grad():
                action = self.model.get_action(
                    preprocessed_frame.squeeze(0),
                    threshold=0.3,
                    use_temporal_smoothing=True
                )

            inference_time = (time.time() - start_time) * 1000

            with self.lock:
                self.stats['frames_processed'] += 1
                self.stats['inference_time_ms'] = inference_time
                self.stats['fps'] = 1.0 / (time.time() - last_inference_time)

            processed_action = self.post_process_action(action)

            try:
                self.action_queue.put(processed_action, timeout=0.01)
            except:
                pass

            self.last_action = processed_action
            last_inference_time = current_time

    def post_process_action(self, action):
        buttons = action['buttons'].cpu().numpy()
        stick = action['stick'].cpu().numpy()

        stick[0] = np.clip(stick[0], -1.0, 1.0) * 127
        stick[1] = np.clip(stick[1], -1.0, 1.0) * 127

        button_mask = buttons > 0
        button_value = sum(2**i for i in range(len(buttons)) if button_mask[i])

        return {
            'buttons': int(button_value),
            'stick_x': int(stick[0]),
            'stick_y': int(stick[1]),
            'button_probs': action['button_probs'].cpu().numpy()
        }

    def start(self):
        if not self.running:
            self.running = True
            self.inference_thread = Thread(target=self.inference_loop, daemon=True)
            self.inference_thread.start()
            print(f"Inference engine started at {self.inference_fps} FPS")

    def stop(self):
        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=1.0)
        print("Inference engine stopped")

    def get_stats(self):
        with self.lock:
            return self.stats.copy()


class FrameInterpolator:
    def __init__(self, interpolation_frames=2):
        self.interpolation_frames = interpolation_frames
        self.last_action = None
        self.target_action = None
        self.interpolation_step = 0

    def set_target(self, action):
        if self.last_action is None:
            self.last_action = action
            self.target_action = action
        else:
            self.last_action = self.get_current_action()
            self.target_action = action
            self.interpolation_step = 0

    def get_current_action(self):
        if self.last_action is None or self.target_action is None:
            return None

        if self.interpolation_step >= self.interpolation_frames:
            return self.target_action

        alpha = self.interpolation_step / self.interpolation_frames
        interpolated = {
            'buttons': self.target_action['buttons'],
            'stick_x': int(self.last_action['stick_x'] * (1 - alpha) + self.target_action['stick_x'] * alpha),
            'stick_y': int(self.last_action['stick_y'] * (1 - alpha) + self.target_action['stick_y'] * alpha)
        }

        self.interpolation_step += 1
        return interpolated