# src/training/trainer.py

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0

    def train_model(self, num_epochs=25, patience=5):
        since = time.time()
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    dataloader = self.train_loader
                else:
                    self.model.eval()
                    dataloader = self.val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and self.scheduler:
                    self.scheduler.step()

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Deep copy the model
                if phase == 'val':
                    if epoch_acc > self.best_acc:
                        self.best_acc = epoch_acc
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {self.best_acc:.4f}')

        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)
        return self.model
