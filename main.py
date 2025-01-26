# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.data_preprocessing import DataPreprocessor
from src.models.cnn_model import CNNModel
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from torchvision import transforms
from src.models.transfer_learning import TransferLearningModel

def main():
    # Configuration
    dataset_name = 'CIFAR10'  # or 'animal'
    data_dir = './data/raw'
    batch_size = 64
    num_epochs = 25
    patience = 5
    learning_rate = 0.001
    model_type = 'cnn'  # or 'transfer_learning'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data Preprocessing
    preprocessor = DataPreprocessor(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        validation_split=0.15,
        test_split=0.15
    )
    train_loader, val_loader, test_loader = preprocessor.get_data_loaders()

    # Model Initialization
    if model_type == 'cnn':
        model = CNNModel(num_classes=10)
    elif model_type == 'transfer_learning':
        model = TransferLearningModel(model_name='resnet18', num_classes=10, feature_extract=True)
    else:
        raise ValueError("Unsupported model type.")

    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training
    trainer = Trainer(model, device, train_loader, val_loader, criterion, optimizer, scheduler)
    best_model = trainer.train_model(num_epochs=num_epochs, patience=patience)

    # Save the best model
    torch.save(best_model.state_dict(), 'best_model.pth')

    # Evaluation
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    evaluator = Evaluator(best_model, device, test_loader, class_names)
    evaluator.evaluate()

if __name__ == '__main__':
    main()
