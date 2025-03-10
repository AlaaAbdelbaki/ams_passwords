import logging
import torch
import torch.nn as nn
from src import device  # Ensure device is imported correctly

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RNNLight(nn.Module):
    """
    A lightweight RNN model for sequence processing, capable of bidirectional processing.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The number of output classes or features.
        bidirectional (bool): Whether the RNN should be bidirectional.
    """
    def __init__(self, input_size, hidden_size, output_size, bidirectional=True):
        super(RNNLight, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = 1  # Fixed n_layers reference
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        
        # Define the RNN layer
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        
        # Output layer
        self.out = nn.Linear(self.num_directions * self.hidden_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Softmax for output probability distribution
        self.softmax = nn.LogSoftmax(dim=1)
        
        logging.info("RNNLight initialized successfully.")

    def forward(self, input, hidden):
        """
        Forward pass for the RNN model.

        Args:
            input (Tensor): The input sequence to the RNN.
            hidden (Tensor): The hidden state from the previous time step.

        Returns:
            output (Tensor): The output prediction.
            hidden (Tensor): The hidden state for the next time step.
        """
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concatenated = torch.cat((hidden[0], hidden[1]), 1) if self.bidirectional else hidden.squeeze(0)
        output = self.out(hidden_concatenated)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Initialize the hidden state as zeros.

        Returns:
            Tensor: The initial hidden state.
        """
        return torch.zeros(self.num_directions, 1, self.hidden_size).to(device)

    def init_hidden_random(self):
        """
        Initialize the hidden state with random values.

        Returns:
            Tensor: The initial hidden state with random values.
        """
        return torch.rand(self.num_directions, 1, self.hidden_size).to(device)
    

class RNN(nn.Module):
    """
    A standard RNN model for sequence processing with multiple layers.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of features in the hidden state.
        output_size (int): The number of output classes or features.
        n_layers (int): The number of layers in the RNN.
    """
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Input to hidden layer
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Input to output layer
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
        # Hidden to hidden output layer
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Softmax for output probability distribution
        self.softmax = nn.LogSoftmax(dim=1)
        
        logging.info("RNN initialized successfully.")

    def forward(self, input, hidden):
        """
        Forward pass for the RNN model.

        Args:
            input (Tensor): The input sequence to the RNN.
            hidden (Tensor): The hidden state from the previous time step.

        Returns:
            output (Tensor): The output prediction.
            hidden (Tensor): The hidden state for the next time step.
        """
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Initialize the hidden state as zeros.

        Returns:
            Tensor: The initial hidden state.
        """
        return torch.zeros(1, self.hidden_size).to(device)

    def init_hidden_random(self):
        """
        Initialize the hidden state with random values.

        Returns:
            Tensor: The initial hidden state with random values.
        """
        return torch.rand(1, self.hidden_size).to(device)
