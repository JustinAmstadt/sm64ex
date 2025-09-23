import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class ModelEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def evaluate_accuracy(self, dataloader):
        button_correct = 0
        stick_errors = []
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                buttons = batch['buttons'].to(self.device)
                stick = batch['stick'].to(self.device)

                output = self.model(images)

                button_preds = (torch.sigmoid(output['buttons']) > 0.5).float()
                button_correct += (button_preds == buttons).all(dim=1).sum().item()

                stick_error = torch.abs(output['stick'] - stick)
                stick_errors.extend(stick_error.cpu().numpy())

                total_samples += images.size(0)

        button_accuracy = button_correct / total_samples
        avg_stick_error = np.mean(stick_errors)

        return {
            'button_accuracy': button_accuracy,
            'avg_stick_error': avg_stick_error,
            'stick_error_std': np.std(stick_errors)
        }

    def visualize_predictions(self, dataloader, num_samples=8):
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 8))
        axes = axes.flatten()

        sample_idx = 0
        with torch.no_grad():
            for batch in dataloader:
                if sample_idx >= num_samples:
                    break

                images = batch['image'].to(self.device)
                buttons = batch['buttons'].to(self.device)
                stick = batch['stick'].to(self.device)

                output = self.model(images)

                for i in range(min(images.size(0), num_samples - sample_idx)):
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                    img = np.clip(img, 0, 1)

                    axes[sample_idx].imshow(img)

                    pred_buttons = (torch.sigmoid(output['buttons'][i]) > 0.5).cpu().numpy()
                    true_buttons = buttons[i].cpu().numpy()
                    pred_stick = output['stick'][i].cpu().numpy()
                    true_stick = stick[i].cpu().numpy()

                    button_acc = (pred_buttons == true_buttons).mean()
                    stick_error = np.abs(pred_stick - true_stick).mean()

                    axes[sample_idx].set_title(f"Button Acc: {button_acc:.2f}\nStick Err: {stick_error:.2f}")
                    axes[sample_idx].axis('off')

                    sample_idx += 1

        plt.tight_layout()
        return fig


class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, loss, metrics=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics
        }

        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path, model, optimizer=None):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('epoch', 0), checkpoint.get('loss', 0), checkpoint.get('metrics', {})

    def get_best_checkpoint(self):
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            return best_path

        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pt'))
        if checkpoints:
            return max(checkpoints, key=lambda p: p.stat().st_mtime)

        return None


class ActionDecoder:
    BUTTON_MAP = {
        0x0001: 'A',
        0x0002: 'B',
        0x0004: 'Start',
        0x0008: 'L',
        0x0010: 'R',
        0x0020: 'Z',
        0x0040: 'C_Up',
        0x0080: 'C_Down',
        0x0100: 'C_Left',
        0x0200: 'C_Right',
        0x0400: 'D_Up',
        0x0800: 'D_Down',
        0x1000: 'D_Left',
        0x2000: 'D_Right'
    }

    @staticmethod
    def decode_buttons(button_value):
        pressed = []
        for mask, name in ActionDecoder.BUTTON_MAP.items():
            if button_value & mask:
                pressed.append(name)
        return pressed

    @staticmethod
    def encode_buttons(button_list):
        value = 0
        for button in button_list:
            for mask, name in ActionDecoder.BUTTON_MAP.items():
                if name == button:
                    value |= mask
        return value


def compute_dataset_statistics(dataloader):
    button_counts = np.zeros(16)
    stick_values_x = []
    stick_values_y = []

    for batch in dataloader:
        buttons = batch['buttons'].numpy()
        stick = batch['stick'].numpy()

        button_counts += buttons.sum(axis=0)
        stick_values_x.extend(stick[:, 0])
        stick_values_y.extend(stick[:, 1])

    total_samples = len(dataloader.dataset)

    stats = {
        'button_frequencies': button_counts / total_samples,
        'stick_x_mean': np.mean(stick_values_x),
        'stick_x_std': np.std(stick_values_x),
        'stick_y_mean': np.mean(stick_values_y),
        'stick_y_std': np.std(stick_values_y),
        'total_samples': total_samples
    }

    return stats