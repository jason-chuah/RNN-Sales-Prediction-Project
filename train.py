import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from model import RNNModel

input_size = 1
hidden_size = 50
num_layers = 2
output_size = 6
num_epochs = 100
learning_rate = 0.001
sequence_length = 4

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values(by=['store', 'product', 'Date'])

    X = []
    y = []
    for store_product, group in df.groupby(['store', 'product']):
        sales = group['number_sold'].values
        for i in range(len(sales) - sequence_length - output_size + 1):
            X.append(sales[i:i + sequence_length])
            y.append(sales[i + sequence_length:i + sequence_length + output_size])
    return np.array(X), np.array(y)

X, y = load_data('train.csv')
X_train, y_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

model = RNNModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.float32)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'best_model.pth')
print('Model saved as best_model.pth')