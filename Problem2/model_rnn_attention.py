import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Standard Elman RNN cell
# h_t = tanh(Wx + Uh + b)
class VanillaRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(VanillaRNNCell, self).__init__()

        self.hidden_size = hidden_size

        # input contribution
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))

        # recurrent contribution (this carries memory)
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # small uniform init to keep early activations stable
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, x_t, h_prev):
        # x_t: (B, input_size)
        # h_prev: (B, hidden_size)

        ih = x_t @ self.weight_ih.T + self.bias_ih
        hh = h_prev @ self.weight_hh.T + self.bias_hh

        # tanh keeps things bounded across timesteps
        return torch.tanh(ih + hh)


# RNN + causal self-attention on top
# idea: RNN builds sequence features, attention lets each timestep look back
class RNNAttentionModel(nn.Module):

    def __init__(self, vocab_size=55, embed_dim=64, hidden_size=128, num_layers=2, dropout=0.3):
        super(RNNAttentionModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # token -> embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # stacked RNN layers
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_size
            self.rnn_cells.append(VanillaRNNCell(in_dim, hidden_size))

        self.dropout = nn.Dropout(dropout)

        # project hidden states to queries (Q)
        # keys (K) and values (V) are just the hidden states themselves
        self.attn_proj = nn.Linear(hidden_size, hidden_size)

        # final layer takes [RNN output || attention context]
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, teacher_force_ratio=1.0):
        # x: (B, T)
        B, T = x.size()
        device = x.device

        # (B, T, embed_dim)
        emb = self.embedding(x)

        # hidden state per layer
        h = [torch.zeros(B, self.hidden_size, device=device) for _ in range(self.num_layers)]

        outputs = []

        # ---- RNN forward pass ----
        for t in range(T):
            x_t = emb[:, t, :]  # current token embedding

            for l, cell in enumerate(self.rnn_cells):
                h[l] = cell(x_t, h[l])
                x_t = h[l]

                # dropout only between layers
                if l < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            outputs.append(x_t)

        # (B, T, hidden)
        outputs = torch.stack(outputs, dim=1)

        # ---- Attention part ----
        # Q: transformed hidden states
        Q = self.attn_proj(outputs)

        # K, V: raw hidden states
        K = outputs
        V = outputs

        # compute attention scores
        # (B, T, T): each timestep attends to all others
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_size)

        # causal mask → no peeking into future
        # lower triangular matrix
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0)

        scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax over past timesteps
        attn_weights = F.softmax(scores, dim=-1)

        # weighted sum of values → context vector
        # (B, T, hidden)
        context = torch.bmm(attn_weights, V)

        # ---- Combine RNN + attention ----
        # concatenate along feature dim
        combined = torch.cat((outputs, context), dim=2)

        combined = self.dropout(combined)

        # final prediction
        logits = self.fc(combined)  # (B, T, vocab_size)

        return logits