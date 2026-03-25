import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

from utils import NameDataset, collate_fn
from model_vanilla_rnn import VanillaRNN
from model_bilstm import BiLSTMModel
from model_rnn_attention import RNNAttentionModel


# Generic training loop (works for all models)
def train_model(model, dataloader, num_epochs=100, lr=0.003, device='cuda', save_path='checkpoints/model.pth'):
    model = model.to(device)

    # Adam works well for sequence models
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ignore_index=0 → padding tokens don't contribute to loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # make sure checkpoint directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epoch_losses = []

    print(f"Starting training for {model.__class__.__name__}...")

    for epoch in range(num_epochs):
        model.train()  # enable dropout, gradients, etc.
        total_loss = 0

        for batch_idx, (x, y, lengths) in enumerate(dataloader):
            # x: input sequence
            # y: target sequence (shifted version)
            x, y = x.to(device), y.to(device)

            # clear old gradients
            optimizer.zero_grad()

            # forward pass → predict logits for every timestep
            logits = model(x)

            # some models might return extra outputs
            if isinstance(logits, tuple):
                logits = logits[0]

            # reshape:
            # (B, T, vocab) → (B*T, vocab)
            # same for targets → (B*T)
            loss = criterion(
                logits.view(-1, model.vocab_size),
                y.view(-1)
            )

            # backprop
            loss.backward()

            # clip gradients → prevents exploding gradients (common in RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # update parameters
            optimizer.step()

            total_loss += loss.item()

        # average loss over all batches
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        # print occasionally (not every epoch to avoid spam)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # save trained weights
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")

    return epoch_losses


def main():
    parser = argparse.ArgumentParser(description='Train Name Generation Models')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    args = parser.parse_args()

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load dataset
    dataset = NameDataset('TrainingNames.txt')

    # dataloader handles batching + padding via collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Vanilla RNN
    vanilla_model = VanillaRNN(vocab_size=dataset.vocab_size)
    train_model(
        vanilla_model,
        dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path='checkpoints/vanilla_rnn.pth'
    )

    # BiLSTM 
    bilstm_model = BiLSTMModel(vocab_size=dataset.vocab_size)
    train_model(
        bilstm_model,
        dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path='checkpoints/bilstm.pth'
    )

    # RNN + Attention
    attn_model = RNNAttentionModel(vocab_size=dataset.vocab_size)
    train_model(
        attn_model,
        dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path='checkpoints/rnn_attention.pth'
    )


if __name__ == '__main__':
    main()