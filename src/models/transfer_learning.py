# src/models/transfer_learning.py

import torch.nn as nn
from torchvision import models

class TransferLearningModel(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=10, feature_extract=True):
        super(TransferLearningModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.model = self.initialize_model()

    def initialize_model(self):
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            self.set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        elif self.model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True)
            self.set_parameter_requires_grad(model, self.feature_extract)
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)
        else:
            raise ValueError("Unsupported model architecture.")

        return model

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)
