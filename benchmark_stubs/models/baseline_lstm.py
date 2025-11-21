import numpy as np
import torch
import torch.nn as nn


class SimpleLSTM:
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1,
                 sequence_length=10, learning_rate=0.001, epochs=50):
        """
        Initialize LSTM model
        :param input_size: Dimension of input features
        :param hidden_size: Size of LSTM hidden layer
        :param num_layers: Number of LSTM layers
        :param output_size: Dimension of output
        :param sequence_length: Length of time sequence
        :param learning_rate: Learning rate for optimizer
        :param epochs: Number of training epochs
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Define LSTM network architecture
        self.model = self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _build_model(self):
        """Build LSTM network architecture"""

        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMNet, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                # x shape: (batch, seq_len, input_size)
                lstm_out, _ = self.lstm(x)
                # Extract output from the last time step
                out = self.fc(lstm_out[:, -1, :])
                return out

        return LSTMNet(self.input_size, self.hidden_size, self.num_layers, self.output_size)

    def _prepare_sequences(self, X, y=None):
        """
        Convert data into time series format
        :param X: Feature data (n_samples, n_features)
        :param y: Target values (n_samples,) - optional
        :return: Sequenced data
        """
        sequences = []
        targets = []

        # Apply padding if number of samples is less than sequence_length
        if X.shape[0] < self.sequence_length:
            padding = np.zeros((self.sequence_length - X.shape[0], X.shape[1]))
            X = np.vstack([padding, X])
            if y is not None:
                y_padding = np.zeros(self.sequence_length - len(y))
                y = np.concatenate([y_padding, y])

        # Create sliding window sequences
        for i in range(len(X) - self.sequence_length + 1):
            seq = X[i:i + self.sequence_length]
            sequences.append(seq)
            if y is not None:
                targets.append(y[i + self.sequence_length - 1])

        sequences = np.array(sequences)

        if y is not None:
            targets = np.array(targets)
            return sequences, targets
        else:
            return sequences

    def fit(self, X, y):
        """
        Train the model
        :param X: Feature data (n_samples, n_features)
        :param y: Target values (n_samples,)
        """
        # Prepare sequence data
        X_seq, y_seq = self._prepare_sequences(X, y)

        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1)

        # Train the model
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        """
        Make predictions using the trained model
        :param X: Feature data (n_samples, n_features)
        :return: Predicted values (n_samples,)
        """
        # Prepare sequence data
        X_seq = self._prepare_sequences(X)

        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X_seq)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.numpy().flatten()