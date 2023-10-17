import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class NeuralNetwork(nn.Module):
    def __init__(self, class_cnt, input_channels):
        super().__init__()
        self.class_cnt = class_cnt
        self.input_channels = input_channels
        self.net_stack = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            nn.Linear(1600, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, class_cnt)           
        )
        
    def forward(self, x):
        logits = self.net_stack(x)
        return logits

def main():
    data_transform = v2.Compose([
        v2.ToImageTensor(),
        v2.ConvertImageDtype()
    ])
    
    class_cnt = 10
    input_channels = 3
    training_dataset = datasets.CIFAR10("data", train=True,
                                     download=True,
                                     transform=data_transform)
    testing_dataset = datasets.CIFAR10("data", train=False,
                                    download=True,
                                    transform=data_transform)
    
    batch_size = 64
    train_dataloader = DataLoader(training_dataset,
                                  batch_size=batch_size)
    test_dataloader = DataLoader(testing_dataset,
                                 batch_size=batch_size)
    
    model = NeuralNetwork(class_cnt, input_channels)
    
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    model = model.to(device)
    print("Using device", device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    total_epochs = 10
    
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X,y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                loss = loss.item()
                current = (batch+1)*len(X)
                print("Loss:", loss, 
                      " at", current, 
                      "of", size)
                
    def test(dataloader, model, loss_fn, data_name):
        size = len(dataloader.dataset)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for X,y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                
                
                
    for epoch in range(total_epochs):
        print("** EPOCH", (epoch+1), "***********")
        train(train_dataloader, model, loss_fn, optimizer)
        
            

if __name__ == "__main__":
    main()
    
