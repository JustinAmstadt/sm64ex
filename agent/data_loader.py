import csv
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class SM64Dataset(Dataset):
    def __init__(self, data_dir='data_store/', transform=None, frame_skip=1):
        self.data_dir = Path(data_dir)
        self.frame_skip = frame_skip
        self.transform = transform if transform else self.get_default_transform()

        self.samples = []
        self._load_data()

    def get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_data(self):
        csv_files = list(self.data_dir.glob('*.csv'))

        for csv_file in csv_files:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                for i in range(0, len(rows), self.frame_skip):
                    row = rows[i]
                    # The CSV already contains the full relative path
                    framebuffer_path = Path(row['Framebuffer_Path'])

                    # If it's not absolute, make it relative to current directory
                    if not framebuffer_path.is_absolute():
                        framebuffer_path = Path.cwd() / framebuffer_path

                    if framebuffer_path.exists():
                        buttons = int(row['Buttons'], 16)
                        stick_x = int(row['Stick_X'])
                        stick_y = int(row['Stick_Y'])

                        self.samples.append({
                            'framebuffer_path': framebuffer_path,
                            'buttons': buttons,
                            'stick_x': stick_x,
                            'stick_y': stick_y
                        })

    def _load_framebuffer(self, path):
        with open(path, 'rb') as f:
            data = f.read()

        width, height = 320, 240

        # Check different possible formats
        if len(data) == width * height * 3:
            # Standard RGB format
            pixels = struct.unpack(f'{width*height*3}B', data)
            image_array = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
        elif len(data) == 841 * 841 * 3:
            # 841x841 RGB format - resize to 320x240
            pixels = struct.unpack(f'{841*841*3}B', data)
            image_array = np.array(pixels, dtype=np.uint8).reshape((841, 841, 3))
            img = Image.fromarray(image_array, mode='RGB')
            img = img.resize((width, height), Image.LANCZOS)
            return img
        elif len(data) > width * height * 3:
            # Try extracting RGB from padded format (every 4th byte pattern)
            # Based on observed pattern: 00 RR 00 00 GG 00 00 BB 00 00...
            extracted = []
            i = 0
            while i < len(data) and len(extracted) < width * height * 3:
                # Skip first byte (padding), take second byte (color value)
                if i + 1 < len(data):
                    extracted.append(data[i + 1])
                i += 4 if len(extracted) % 3 == 0 else 1

            if len(extracted) >= width * height * 3:
                pixels = extracted[:width * height * 3]
                image_array = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
            else:
                # Fallback: treat as unknown large format, extract what we can
                # Take every 4th byte starting from position 1
                pixels = []
                for j in range(1, min(len(data), width * height * 3 * 4), 4):
                    pixels.append(data[j])

                if len(pixels) >= width * height * 3:
                    pixels = pixels[:width * height * 3]
                else:
                    # Pad with zeros if needed
                    pixels.extend([0] * (width * height * 3 - len(pixels)))

                image_array = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
        else:
            raise ValueError(f"Framebuffer size {len(data)} not supported")

        # Ensure we return a PIL Image, converting numpy array properly
        return Image.fromarray(np.uint8(image_array), mode='RGB')

    def _buttons_to_binary(self, buttons):
        binary = np.zeros(16, dtype=np.float32)
        for i in range(16):
            if buttons & (1 << i):
                binary[i] = 1.0
        return binary

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = self._load_framebuffer(sample['framebuffer_path'])
        image = self.transform(image)

        buttons = self._buttons_to_binary(sample['buttons'])
        stick = np.array([sample['stick_x'] / 127.0, sample['stick_y'] / 127.0], dtype=np.float32)

        return {
            'image': image,
            'buttons': torch.FloatTensor(buttons),
            'stick': torch.FloatTensor(stick)
        }


def get_dataloader(batch_size=32, shuffle=True, num_workers=4, frame_skip=1):
    dataset = SM64Dataset(frame_skip=frame_skip)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                     num_workers=num_workers, pin_memory=True)