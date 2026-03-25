import torch
import torch.nn as nn
import math


# Single LSTM cell (manual implementation)
# Keeps both hidden state (h) and cell state (c)
class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()

        self.hidden_size = hidden_size

        # One big weight matrix for all 4 gates:
        # input (i), forget (f), candidate (g), output (o)
        # shape: (4 * hidden_size, input_size)
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size * 4, input_size))

        # recurrent weights
        # shape: (4 * hidden_size, hidden_size)
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size * 4, hidden_size))

        # biases for input and hidden parts
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size * 4))

        self.reset_parameters()

    def reset_parameters(self):
        # same idea as before: keep things stable at the start
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, x, hx):
        # x: (B, input_size)
        # hx: tuple (h_prev, c_prev), both (B, hidden_size)
        h_prev, c_prev = hx

        # compute all gate activations in one go (more efficient)
        gates = (
            x @ self.weight_ih.T + self.bias_ih +
            h_prev @ self.weight_hh.T + self.bias_hh
        )

        # split into 4 parts along feature dimension
        i, f, g, o = gates.chunk(4, dim=1)

        # apply non-linearities
        i = torch.sigmoid(i)      # how much new info to write
        f = torch.sigmoid(f)      # how much old memory to keep
        g = torch.tanh(g)         # candidate memory
        o = torch.sigmoid(o)      # how much to expose as output

        # update cell state (this is the key LSTM idea)
        c_next = f * c_prev + i * g

        # hidden state is filtered version of cell state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# Bidirectional multi-layer LSTM
class BiLSTMModel(nn.Module):

    def __init__(self, vocab_size=55, embed_dim=64, hidden_size=128, num_layers=2, dropout=0.3):
        super(BiLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # token -> embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # forward and backward LSTM stacks
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()

        for i in range(num_layers):
            # after first layer, input is concatenated (forward + backward)
            in_dim = embed_dim if i == 0 else hidden_size * 2

            self.forward_cells.append(LSTMCell(in_dim, hidden_size))
            self.backward_cells.append(LSTMCell(in_dim, hidden_size))

        self.dropout = nn.Dropout(dropout)

        # final projection uses both directions → 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # x: (B, T)
        B, T = x.size()
        device = x.device

        # (B, T, embed_dim)
        emb = self.embedding(x)

        # initialize (h, c) for each layer
        states_f = [
            (torch.zeros(B, self.hidden_size, device=device),
             torch.zeros(B, self.hidden_size, device=device))
            for _ in range(self.num_layers)
        ]

        states_b = [
            (torch.zeros(B, self.hidden_size, device=device),
             torch.zeros(B, self.hidden_size, device=device))
            for _ in range(self.num_layers)
        ]

        forward_outs = []
        backward_outs = []

        # ---- forward pass (left → right) ----
        for t in range(T):
            x_t = emb[:, t, :]

            for l, cell in enumerate(self.forward_cells):
                h, c = cell(x_t, states_f[l])
                states_f[l] = (h, c)
                x_t = h

                if l < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            forward_outs.append(x_t)

        # ---- backward pass (right → left) ----
        for t in range(T):
            x_t = emb[:, T - 1 - t, :]

            for l, cell in enumerate(self.backward_cells):
                h, c = cell(x_t, states_b[l])
                states_b[l] = (h, c)
                x_t = h

                if l < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            backward_outs.append(x_t)

        # reverse backward outputs so they align with forward time
        backward_outs = backward_outs[::-1]

        # ---- combine both directions ----
        combined = []
        for t in range(T):
            # concat forward + backward features
            combined.append(torch.cat((forward_outs[t], backward_outs[t]), dim=1))

        # (B, T, 2 * hidden_size)
        outs = torch.stack(combined, dim=1)

        outs = self.dropout(outs)

        # final token predictions
        logits = self.fc(outs)

        return logits