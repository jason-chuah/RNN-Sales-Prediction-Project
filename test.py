import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from model import RNNModel
import torch

input_size = 1
hidden_size = 50
num_layers = 2
output_size = 6

df = pd.read_csv("test_example.csv", parse_dates=['Date'])
model = RNNModel(input_size, hidden_size, num_layers, output_size)
window_size = 10

gb = df.groupby(["store", "product"])
groups = {x: gb.get_group(x) for x in gb.groups}
scores = {}

for key, data in groups.items():
    X = data['number_sold'].values
    N = X.shape[0]

    mape_score = []
    start = window_size
    while start + 6 <= N:
        inputs = X[(start - window_size) : start]
        targets = X[start : (start + 6)]
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(-1)
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        with torch.no_grad():
          predictions = model(inputs.unsqueeze(0)).squeeze(0).numpy()
        start += 6
        mape_score.append(mean_absolute_percentage_error(targets, predictions))
    scores[key] = mape_score

np.savez("score.npz", scores=scores)