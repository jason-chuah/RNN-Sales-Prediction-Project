## Predicting Future Sales Using Recurrent Neural Networks
<br></br>
### Objective
The goal of this project is to develop a Recurrent Neural Network (RNN) model to predict
future sales based on historical sales data. The model uses a sequence of past sales data to
predict sales for the next five time steps.
<br></br>
### Model Architecture
The RNN model is implemented using PyTorch and has the following architecture:
- Input Layer: Takes the sequence of sales data.
- Hidden Layers: Two LSTM layers with 50 hidden units each.
- Output Layer: A fully connected layer that outputs predictions for the next five time
steps.
<br></br>
### Training Procedure
1. Data Loading and Preprocessing:
- The training data is loaded and sorted by store, product, and Date.
- Sequences of 4 days are used as input to predict sales for the next 5 days.
- Data is converted to PyTorch tensors.
2. Model Initialization:
- The RNN model is initialized with input size 1, hidden size 50, 2 layers, and
output size 5.
- The model is trained using the Adam optimizer and Mean Squared Error
(MSE) loss function.
3. Training Loop:
- The model is trained for 100 epochs with a batch size of 32.
- The loss is printed every 10 epochs.
- The trained model is saved to best_model.pth.
<br></br>
### Results
The model was trained for 100 epochs, and the Mean Squared Error (MSE) loss was
monitored every 10 epochs. The training loss values fluctuated over the training period,
indicating significant variability during training. This suggests that the model might have had
difficulties consistently capturing the patterns in the data. This fluctuation could be due to
various factors such as the complexity of the data, the model architecture, or the chosen
hyperparameters.
Despite the variability in training loss, the model was able to learn from the data to some
extent and produce predictions for the test dataset (test_example.csv). The results highlight
that while the model managed to capture some temporal dependencies in the sales data,
further refinement and tuning would be necessary to improve the model's stability and
accuracy in forecasting future sales.
<br></br>
### Conclusion
While the current RNN model demonstrates some predictive capability, there is substantial
room for improvement. Further refinement and experimentation are necessary to achieve a
more robust and accurate sales forecasting model.