import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter  # Disabled due to numpy compatibility
from tqdm import tqdm
from pathlib import Path
from datetime import datetime


class SM64Trainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

        self.criterion_buttons = nn.BCEWithLogitsLoss()
        self.criterion_stick = nn.MSELoss()

        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        self.best_loss = float('inf')
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # self.writer = SummaryWriter(f'runs/sm64_agent_{timestamp}')  # Disabled due to numpy compatibility
        self.writer = None

    def compute_loss(self, output, target):
        button_loss = self.criterion_buttons(output['buttons'], target['buttons'])
        stick_loss = self.criterion_stick(output['stick'], target['stick'])

        total_loss = button_loss + 2.0 * stick_loss

        return {
            'total': total_loss,
            'buttons': button_loss,
            'stick': stick_loss
        }

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        button_loss_sum = 0
        stick_loss_sum = 0
        num_batches = 0

        progress = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in progress:
            images = batch['image'].to(self.device)
            buttons = batch['buttons'].to(self.device)
            stick = batch['stick'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(images)

            losses = self.compute_loss(output, {'buttons': buttons, 'stick': stick})
            losses['total'].backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += losses['total'].item()
            button_loss_sum += losses['buttons'].item()
            stick_loss_sum += losses['stick'].item()
            num_batches += 1

            progress.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'button': f"{losses['buttons'].item():.4f}",
                'stick': f"{losses['stick'].item():.4f}"
            })

        avg_loss = total_loss / num_batches
        avg_button_loss = button_loss_sum / num_batches
        avg_stick_loss = stick_loss_sum / num_batches

        return avg_loss, avg_button_loss, avg_stick_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        button_accuracy = 0
        stick_error = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                images = batch['image'].to(self.device)
                buttons = batch['buttons'].to(self.device)
                stick = batch['stick'].to(self.device)

                output = self.model(images)
                losses = self.compute_loss(output, {'buttons': buttons, 'stick': stick})

                total_loss += losses['total'].item()

                button_preds = (torch.sigmoid(output['buttons']) > 0.5).float()
                button_accuracy += (button_preds == buttons).float().mean().item()

                stick_error += torch.abs(output['stick'] - stick).mean().item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_button_acc = button_accuracy / num_batches
        avg_stick_error = stick_error / num_batches

        return avg_loss, avg_button_acc, avg_stick_error

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss: {loss:.4f}")

        last_path = self.checkpoint_dir / 'last_checkpoint.pt'
        torch.save(checkpoint, last_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def train(self, train_loader, val_loader=None, num_epochs=100, save_freq=5):
        for epoch in range(1, num_epochs + 1):
            train_loss, train_button_loss, train_stick_loss = self.train_epoch(train_loader, epoch)

            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Train_Buttons', train_button_loss, epoch)
                self.writer.add_scalar('Loss/Train_Stick', train_stick_loss, epoch)
                self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            if val_loader:
                val_loss, val_button_acc, val_stick_error = self.validate(val_loader)
                print(f"Validation: Loss = {val_loss:.4f}, Button Acc = {val_button_acc:.4f}, Stick Error = {val_stick_error:.4f}")

                if self.writer:
                    self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                    self.writer.add_scalar('Accuracy/Buttons', val_button_acc, epoch)
                    self.writer.add_scalar('Error/Stick', val_stick_error, epoch)

                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
            else:
                is_best = train_loss < self.best_loss
                if is_best:
                    self.best_loss = train_loss

            if epoch % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, train_loss, is_best)

            self.scheduler.step()

        if self.writer:
            self.writer.close()
        print(f"Training complete! Best loss: {self.best_loss:.4f}")