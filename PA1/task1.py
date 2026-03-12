import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class GloVeTorch(nn.Module):
    def __init__(self, vocab_size, embedding_dim, x_max=100, alpha=0.75):
        super().__init__()

        self.V = vocab_size
        self.d = embedding_dim
        self.x_max = x_max
        self.alpha = alpha

        # Word embeddings
        self.W = nn.Embedding(self.V, self.d)
        self.W_tilde = nn.Embedding(self.V, self.d)

        # Bias terms
        self.b = nn.Embedding(self.V, 1)
        self.b_tilde = nn.Embedding(self.V, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_tilde.weight)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b_tilde.weight)

    def weighting_function(self, x):
        # vectorized version
        weights = torch.where(
            x < self.x_max,
            (x / self.x_max) ** self.alpha,
            torch.ones_like(x)
        )
        return weights

    def forward(self, i_idx, j_idx, x_ij):

        w_i = self.W(i_idx)                # (batch, d)
        w_j = self.W_tilde(j_idx)          # (batch, d)

        b_i = self.b(i_idx).squeeze()      # (batch,)
        b_j = self.b_tilde(j_idx).squeeze()

        dot = torch.sum(w_i * w_j, dim=1)

        log_x = torch.log(x_ij)

        weights = self.weighting_function(x_ij)

        loss = weights * (dot + b_i + b_j - log_x) ** 2

        return torch.mean(loss)

    def get_embeddings(self):
        return self.W.weight.data + self.W_tilde.weight.data


def train_glove(model, cooc_triples, epochs=10, batch_size=1024, lr=0.001, device="cpu"):

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert triples to tensors
    i_idx = torch.tensor([t[0] for t in cooc_triples], dtype=torch.long)
    j_idx = torch.tensor([t[1] for t in cooc_triples], dtype=torch.long)
    x_ij = torch.tensor([t[2] for t in cooc_triples], dtype=torch.float32)

    dataset_size = len(cooc_triples)

    losses = []

    for epoch in range(epochs):

        permutation = torch.randperm(dataset_size)

        total_loss = 0

        for k in tqdm(range(0, dataset_size, batch_size)):

            indices = permutation[k:k+batch_size]

            batch_i = i_idx[indices].to(device)
            batch_j = j_idx[indices].to(device)
            batch_x = x_ij[indices].to(device)

            optimizer.zero_grad()

            loss = model(batch_i, batch_j, batch_x)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (dataset_size // batch_size + 1)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    return losses


import json

# Load triples
with open("cooc_matrix.json", "r") as f:
    cooc_triples = json.load(f)

with open("vocab_word2idx.json", "r") as f:
    vocab = json.load(f)

vocab_size = len(vocab)
embedding_dim = 200

model = GloVeTorch(vocab_size, embedding_dim)

losses = train_glove(
    model,
    cooc_triples,
    epochs=20,
    batch_size=4096,
    lr=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

embeddings = model.get_embeddings()
torch.save(embeddings.cpu(), "glove_embeddings.pt")
