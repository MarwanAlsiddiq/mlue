import torch
import torch.nn as nn
from torchvision import models

class InceptionTradingModel:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        
    def build_model(self):
        """Build Inception V3 model for trading pattern recognition"""
        # Load pretrained InceptionV3
        model = models.inception_v3(weights='IMAGENET1K_V1')
        
        # Freeze the base model
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes)
        )
        
        return model
        
    def get_optimizer(self, model, learning_rate=0.001):
        """Get optimizer for the model"""
        return torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
        
    def get_criterion(self):
        """Get loss function"""
        return nn.CrossEntropyLoss()
