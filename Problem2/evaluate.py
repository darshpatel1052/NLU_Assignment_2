import torch
import torch.nn.functional as F
import random
import argparse
from tqdm import tqdm

from utils import NameDataset
from model_vanilla_rnn import VanillaRNN
from model_bilstm import BiLSTMModel
from model_rnn_attention import RNNAttentionModel


# Generate a name character-by-character (autoregressive decoding)
def generate_name(model, dataset, max_len=30, temperature=1.0, device='cuda'):
    model.eval()  # turn off dropout etc.

    # start from <SOS> token
    current_idx = dataset.SOS
    generated_indices = []

    with torch.no_grad():
        # model expects a sequence → start with just SOS
        input_seq = torch.tensor([[current_idx]], dtype=torch.long).to(device)

        for _ in range(max_len):
            # run full sequence through model
            logits = model(input_seq)

            # some models might return extra stuff
            if isinstance(logits, tuple):
                logits = logits[0]

            # take only last timestep prediction
            # (predict next character given full prefix)
            next_token_logits = logits[0, -1, :] / temperature

            # convert logits → probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # sample instead of argmax → gives diversity
            next_idx = torch.multinomial(probs, num_samples=1).item()

            # stop if <EOS> predicted
            if next_idx == dataset.EOS:
                break

            generated_indices.append(next_idx)

            # append new token to input sequence (autoregressive loop)
            next_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(device)
            input_seq = torch.cat([input_seq, next_tensor], dim=1)

    # convert indices back to characters
    generated_name = ''.join([dataset.idx2char[idx] for idx in generated_indices])

    return generated_name


# Generate many samples and compute simple quality metrics
def evaluate_model(model, dataset, model_name, num_samples=1000, device='cuda'):
    print(f"\nEvaluating {model_name}...")

    generated_names = []

    # generate names one by one
    for _ in tqdm(range(num_samples), desc=f"Generating for {model_name}"):
        name = generate_name(model, dataset, device=device)
        generated_names.append(name)

    # show a few samples (quick qualitative check)
    print(f"\n--- 10 samples from {model_name} ---")
    for i in range(10):
        print(f"{i+1}. {generated_names[i]}")

    # compare with training data
    training_set = set(dataset.names)

    # novelty: how many generated names are NOT in training set
    novel_count = sum(1 for name in generated_names if name not in training_set)
    novelty_rate = (novel_count / num_samples) * 100

    # diversity: how many unique outputs we got
    unique_names = set(generated_names)
    diversity = len(unique_names) / num_samples

    print(f"\n--- Metrics for {model_name} ---")
    print(f"Novelty Rate: {novelty_rate:.2f}%")
    print(f"Diversity:    {diversity:.4f} ({len(unique_names)} unique out of {num_samples})")

    return generated_names


def main():
    parser = argparse.ArgumentParser(description='Evaluate Name Generation Models')
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of names to generate per model')
    args = parser.parse_args()

    # pick GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load dataset (handles vocab + mappings)
    dataset = NameDataset('TrainingNames.txt')

    # Vanilla RNN
    try:
        vanilla = VanillaRNN(vocab_size=dataset.vocab_size).to(device)
        vanilla.load_state_dict(torch.load('checkpoints/vanilla_rnn.pth', map_location=device))

        evaluate_model(vanilla, dataset, "Vanilla RNN",
                       num_samples=args.samples, device=device)

    except FileNotFoundError:
        print("Model vanilla_rnn.pth not found. Train it first.")

    # BiLSTM
    try:
        bilstm = BiLSTMModel(vocab_size=dataset.vocab_size).to(device)
        bilstm.load_state_dict(torch.load('checkpoints/bilstm.pth', map_location=device))

        evaluate_model(bilstm, dataset, "Bidirectional LSTM",
                       num_samples=args.samples, device=device)

    except FileNotFoundError:
        print("Model bilstm.pth not found. Train it first.")

    # RNN + Attention
    try:
        attn = RNNAttentionModel(vocab_size=dataset.vocab_size).to(device)
        attn.load_state_dict(torch.load('checkpoints/rnn_attention.pth', map_location=device))

        evaluate_model(attn, dataset, "RNN with Attention",
                       num_samples=args.samples, device=device)

    except FileNotFoundError:
        print("Model rnn_attention.pth not found. Train it first.")


if __name__ == '__main__':
    main()