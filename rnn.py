import torch
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers = 3):   
        super().__init__() 

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        self.linear = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        #RNN's hidden state dimension is [num_layers, batch_size, hidden_dim]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.rnn(x, h0)
        out = self.linear(out[:, -1, :]).view([-1, self.num_classes])

        return out

def train(model, dataloader, criterion, optimizer, scheduler, epoch):
    model.to(device)
    model.train()

    train_correct = 0
    train_count = 0
    # log_interval = 30

    for index, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        inputs = inputs.view(-1, 32, 96)
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        pred = torch.argmax(outputs, dim=-1)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        train_correct += (pred == labels).sum().item()
        train_count += labels.size(0)
        train_acc = train_correct / train_count

    scheduler.step()

    return train_acc

def validate(model, dataloader, HEIGHT, input_dim):
    with torch.no_grad():
        model.eval()

        val_count = 0
        val_correct = 0

        for index, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(-1, HEIGHT, input_dim)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            predicts = torch.argmax(outs, 1)

            val_correct += (predicts == labels).sum().item()
            val_count += inputs.size(0)

            val_acc = (val_correct / val_count)

    return val_acc
