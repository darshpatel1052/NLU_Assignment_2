import torch
import torch.nn as nn
import math


# Single RNN cell (Elman RNN)
# Computes: h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)
class VanillaRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(VanillaRNNCell, self).__init__()

        self.hidden_size = hidden_size

        # Weight for current input x_t
        # shape: (hidden_size, input_size)
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))

        # Weight for previous hidden state h_{t-1}
        # this is what carries information across time
        # shape: (hidden_size, hidden_size)
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights in a small range so activations don't blow up early
        # same idea as PyTorch default RNN init
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, x_t, h_prev):
        # x_t: (batch_size, input_size)
        # h_prev: (batch_size, hidden_size)

        # Contribution from input at current timestep
        ih = x_t @ self.weight_ih.T + self.bias_ih

        # Contribution from previous hidden state (memory)
        hh = h_prev @ self.weight_hh.T + self.bias_hh

        # Combine both and apply non-linearity
        # tanh keeps values bounded and introduces non-linearity
        h_t = torch.tanh(ih + hh)

        return h_t


# Multi-layer (stacked) Vanilla RNN
class VanillaRNN(nn.Module):

    def __init__(self, vocab_size=55, embed_dim=64, hidden_size=128, num_layers=2, dropout=0.3):
        super(VanillaRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Converts token indices -> dense vectors
        # input: (B, T) → output: (B, T, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Stack multiple RNN layers manually
        # each layer takes output of previous layer as input
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_size
            self.rnn_cells.append(VanillaRNNCell(in_dim, hidden_size))

        # Dropout for regularization
        # applied between layers and before final output
        self.dropout = nn.Dropout(dropout)

        # Final linear layer to map hidden states → vocabulary scores
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        B, T = x.size()
        device = x.device

        # Get embeddings for all timesteps
        # (B, T, embed_dim)
        emb = self.embedding(x)

        # Initialize hidden state for each layer
        # we keep separate hidden states per layer
        h = [
            torch.zeros(B, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        # Iterate over sequence (time dimension)
        for t in range(T):
            # input at timestep t
            x_t = emb[:, t, :]  # (B, embed_dim)

            # pass through each RNN layer
            for l, cell in enumerate(self.rnn_cells):
                # update hidden state of this layer
                h[l] = cell(x_t, h[l])

                # output of this layer becomes input to next layer
                x_t = h[l]

                # apply dropout between layers (not after last layer)
                if l < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            # store output from last layer
            outputs.append(x_t)

        # combine outputs across all timesteps
        # (B, T, hidden_size)
        outputs = torch.stack(outputs, dim=1)

        # optional dropout before final projection
        outputs = self.dropout(outputs)

        # map to vocabulary logits
        # (B, T, vocab_size)
        logits = self.fc(outputs)

        return logits