import torch
from torch import nn, optim, utils

class Trainner():
    def __init__(self, 
            model: nn.Module, 
            train_data: utils.data.DataLoader,
            test_data: utils.data.DataLoader,
            optimizer: nn.Module = optim.AdamW,
            criterion: nn.Module = nn.CrossEntropyLoss,
            batch_size: int = 16,
            num_epochs: int = 200,
            ):
        
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

    def train(self):
        self.model.train()
        train_loss = 0
        for epoch in range(self.num_epochs):
            for i, (inputs, labels) in enumerate(self.train_data):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {train_loss / 100}")
        
        return train_loss / len(self.train_data)
    
    def test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in self.test_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
        
        return test_loss / len(self.test_data)
