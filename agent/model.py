import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights


class SM64Agent(nn.Module):
    def __init__(self, num_buttons=16, hidden_dim=512, dropout=0.1):
        super().__init__()

        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        vit_output_dim = 768

        self.button_head = nn.Sequential(
            nn.Linear(vit_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_buttons)
        )

        self.stick_head = nn.Sequential(
            nn.Linear(vit_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh()
        )

        self.temporal_buffer = None
        self.alpha = 0.7

    def forward(self, x, use_temporal_smoothing=False):
        features = self.vit(x)

        button_logits = self.button_head(features)
        stick_values = self.stick_head(features)

        if use_temporal_smoothing and self.temporal_buffer is not None:
            stick_values = self.alpha * stick_values + (1 - self.alpha) * self.temporal_buffer

        if use_temporal_smoothing:
            self.temporal_buffer = stick_values.detach()

        return {
            'buttons': button_logits,
            'stick': stick_values
        }

    def get_action(self, image, threshold=0.5, use_temporal_smoothing=True):
        with torch.no_grad():
            output = self.forward(image.unsqueeze(0), use_temporal_smoothing)

            button_probs = torch.sigmoid(output['buttons'])
            buttons = (button_probs > threshold).float()

            stick = output['stick'].squeeze(0)

            return {
                'buttons': buttons.squeeze(0),
                'stick': stick,
                'button_probs': button_probs.squeeze(0)
            }


class ActionBuffer:
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.stick_buffer = []
        self.button_buffer = []

    def add(self, buttons, stick):
        self.button_buffer.append(buttons)
        self.stick_buffer.append(stick)

        if len(self.button_buffer) > self.buffer_size:
            self.button_buffer.pop(0)
            self.stick_buffer.pop(0)

    def get_smoothed_action(self):
        if not self.stick_buffer:
            return None, None

        buttons = torch.stack(self.button_buffer).mean(dim=0)
        buttons = (buttons > 0.5).float()

        stick = torch.stack(self.stick_buffer).mean(dim=0)

        return buttons, stick