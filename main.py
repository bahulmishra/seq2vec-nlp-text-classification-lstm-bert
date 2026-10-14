import copy
import os
import re
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Figures are only saved to disk, never shown; works headless and inside DataLoader workers
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

# Set seed for reproducibility across NumPy and PyTorch
SEED = 551
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def getDevice():
    """
    Detects and returns the best available hardware accelerator device.
    Returns torch.device object consistently for tensor device operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = getDevice()

# DataLoader hardware optimization constants
NUM_WORKERS = min(4, os.cpu_count() if os.cpu_count() else 0)
PIN_MEMORY = True if DEVICE.type == "cuda" else False

"""## Task 1: Acquire and Pre-process the Web of Science Dataset"""

DATA_DIR = "WebOfScienceDataset/WOS11967"
GLOVE_DIR = "glove.6B"
FIGURES_DIR = "figures"

MAX_SEQ_LEN = 200  # Default only; the real value is the 95th percentile of training-set token lengths
EMBED_DIM = 300    # 300-dimensional GloVe pre-trained vectors (used by the final LSTM models)
GLOVE_PATH = os.path.join(GLOVE_DIR, f"glove.6B.{EMBED_DIM}d.txt")
MAX_VOCAB = 10000  # Maximum vocabulary size (top 10,000 frequent terms)
BATCH_SIZE = 32

BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 256
BERT_EPOCHS = 3

# Epoch budgets. Sweeps use a smaller budget for the sub-field task to keep total search time manageable;
# the final model gets a few more epochs and keeps its best validation epoch, so the extra epochs cannot hurt it.
LSTM_SWEEP_EPOCHS_DOMAIN = 10
LSTM_SWEEP_EPOCHS_SUB = 20
LSTM_FINAL_EPOCHS_DOMAIN = 10
LSTM_FINAL_EPOCHS_SUB = 25

# Settings fixed by design for the final LSTMs. The init / L1 / GloVe-dimension sweeps are ablations:
# their winners are reported but not applied automatically (single-seed differences are within run-to-run noise).
FINAL_INIT_TYPE = "xavier"
FINAL_L1_LAMBDA = 0.0

def make_loader(dataset, shuffle=False, batch_size=BATCH_SIZE):
    """
    Builds a DataLoader with the shared worker / pinned-memory settings.
    persistent_workers avoids re-spawning worker processes every epoch (expensive on Windows/macOS spawn).
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS > 0)

def clean_text(text):
    """
    Pre-processes text by:
    1. Lowercasing to unify word representations ("Protein" -> "protein")
    2. Stripping non-alphanumeric characters (removes LaTeX/HTML fragments and noise)
    3. Normalizing whitespace (collapses multiple spaces into one)
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data(data_dir, clean=True):
    """
    Loads raw text abstracts (X), sub-field targets (Y), and 7-class parent domain targets (YL1)
    from the dataset directory.
    - clean=True: LSTM-style cleaning (lowercase, punctuation stripped).
    - clean=False: original text with whitespace normalized only (used for BERT's WordPiece tokenizer).
    """
    with open(os.path.join(data_dir, 'X.txt'), 'r', encoding='utf-8') as f:
        texts = [clean_text(line) if clean else " ".join(line.split()) for line in f.readlines()]

    with open(os.path.join(data_dir, 'Y.txt'), 'r', encoding='utf-8') as f:
        y_sub = [int(line.strip()) for line in f.readlines()]

    with open(os.path.join(data_dir, 'YL1.txt'), 'r', encoding='utf-8') as f:
        y_domain = [int(line.strip()) for line in f.readlines()]

    if not (len(texts) == len(y_sub) == len(y_domain)):
        raise ValueError(f"Dataset files are misaligned: {len(texts)} texts, "
                         f"{len(y_sub)} sub-field labels, {len(y_domain)} domain labels.")

    return texts, y_sub, y_domain

def get_dynamic_max_len(texts, percentile=95):
    """
    Calculates the sequence length covering `percentile`% of all input documents.
    This dynamically balances context retention against wasteful zero-padding memory overhead.
    """
    lengths = [len(t.split()) for t in texts]
    limit = int(np.percentile(lengths, percentile))
    print(f"{percentile}th percentile length is {limit}. Mean is {int(np.mean(lengths))}.")
    return limit

def build_vocab(texts, max_words=MAX_VOCAB, min_freq=2):
    """
    Constructs integer vocabulary mapping for the top `max_words` most frequent terms.
    - Token <PAD> (index 0): Used for sequence padding; setting padding_idx=0 in nn.Embedding
      zeroes out its initial embedding and gradient updates.
    - Token <UNK> (index 1): Fallback for out-of-vocabulary terms or rare tokens (freq < min_freq).
    - min_freq=2: Filters hapax legomena (singletons) to prevent vocabulary over-fitting.
    """
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, count in word_counts.most_common(max_words - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab

def encode_texts(texts, vocab, max_len=MAX_SEQ_LEN):
    """
    Converts list of raw text strings into a 2D LongTensor of word indices [N, max_len].
    - Sequences shorter than max_len are right-padded with <PAD> (0).
    - Sequences longer than max_len are truncated from the right.
    """
    tensor_data = []
    unk_idx = vocab["<UNK>"]
    pad_idx = vocab["<PAD>"]

    for text in texts:
        tokens = text.split()
        seq = [vocab.get(t, unk_idx) for t in tokens]

        if len(seq) < max_len:
            seq = seq + [pad_idx] * (max_len - len(seq))
        else:
            seq = seq[:max_len]

        tensor_data.append(seq)

    return torch.tensor(tensor_data, dtype=torch.long)

def load_glove_matrix(path, vocab, embed_dim=EMBED_DIM):
    """
    Loads pre-trained GloVe vectors into a 2D float tensor [vocab_size, embed_dim].
    - O(1) set lookups prevent loading unwanted vectors into memory.
    - Missing vocabulary tokens are initialized uniformly in [-0.25, 0.25].
    - Token <PAD> (index 0) is explicitly assigned an all-zero vector.
    """
    weights = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim))

    if "<PAD>" in vocab:
        weights[vocab["<PAD>"]] = 0.0

    hits = 0
    vocab_set = set(vocab.keys())
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]

            if word in vocab_set:
                vector = np.array(values[1:], dtype=float)
                if len(vector) == embed_dim:
                    weights[vocab[word]] = vector
                    hits += 1

    print(f"GloVe loaded. Found {hits} / {len(vocab)} words.")
    return torch.tensor(weights, dtype=torch.float32)

def split_indices(n):
    """
    Seeded 64/16/20 Train/Validation/Test split of document indices.
    The same indices are used for the LSTM and BERT so both models see identical examples.
    """
    test_size = int(0.2 * n)
    remaining_size = n - test_size
    val_size = int(0.2 * remaining_size)
    train_size = remaining_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_idx, val_idx, test_idx = random_split(range(n), [train_size, val_size, test_size], generator=generator)
    return list(train_idx), list(val_idx), list(test_idx)

def prepare_data():
    """
    Splits the dataset first, then builds the vocabulary and sequence length from the training split only
    (so no test-set statistics leak into preprocessing), and loads the GloVe embedding matrix.
    """
    texts, y_sub, y_domain = load_data(DATA_DIR)
    train_idx, val_idx, test_idx = split_indices(len(texts))

    train_texts = [texts[i] for i in train_idx]
    dynamic_max_len = get_dynamic_max_len(train_texts)
    vocab = build_vocab(train_texts)

    X_tensor = encode_texts(texts, vocab, max_len=dynamic_max_len)
    Y_sub_tensor = torch.tensor(y_sub, dtype=torch.long)
    Y_domain_tensor = torch.tensor(y_domain, dtype=torch.long)

    embedding_weights = load_glove_matrix(GLOVE_PATH, vocab)

    dataset = TensorDataset(X_tensor, Y_domain_tensor, Y_sub_tensor)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset, embedding_weights, vocab

"""## Task 2: Implement LSTM and BERT models"""

class CustomLSTMCell(nn.Module):
    """
    From-scratch LSTM Cell implementing gating mathematics manually:
    - i_t = σ(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)     [Input gate]
    - f_t = σ(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)     [Forget gate]
    - g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)  [Candidate cell state]
    - o_t = σ(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)     [Output gate]
    - c_t = f_t * c_{t-1} + i_t * g_t                        [Cell state update]
    - h_t = o_t * tanh(c_t)                                  [Hidden state output]
    """
    def __init__(self, input_size, hidden_size, init_type):
        super(CustomLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # Fused linear projections for all 4 gates (4 * hidden_size) for matrix efficiency
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)

        # Weight Initializations:
        # - Xavier Uniform: Preserves variance of activations across deep sequence steps.
        # - Zero Init: Sets initial weights to 0 (tested as benchmark baseline).
        # - Random Normal: Normal distribution N(0, 0.01).
        if init_type == 'xavier':
            nn.init.xavier_uniform_(self.weight_ih.weight)
            nn.init.xavier_uniform_(self.weight_hh.weight)
        elif init_type == 'zero':
            nn.init.zeros_(self.weight_ih.weight)
            nn.init.zeros_(self.weight_hh.weight)
        elif init_type == 'random':
            nn.init.normal_(self.weight_ih.weight, mean=0, std=0.01)
            nn.init.normal_(self.weight_hh.weight, mean=0, std=0.01)
        else:
            raise ValueError(f"Unknown init_type: {init_type!r}")

    def forward(self, x, state):
        h_prev, c_prev = state

        # Single fused matrix multiplication for all 4 gates
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        c_next = (f * c_prev) + (i * g)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class CustomLSTMModel(nn.Module):
    """
    Sequence-to-Vector (Seq2Vec) LSTM Text Classifier using CustomLSTMCell.
    Includes token masking to freeze (h_t, c_t) state updates on padding tokens (<PAD> = 0),
    preventing trailing padding steps from corrupting final document hidden state representations.
    `task` selects the target column of (X, Y_domain, Y_sub) batches: "domain" or "sub".
    """
    TARGET_INDEX = {"domain": 1, "sub": 2}

    def __init__(self, vocab_size, embed_dim, output_dim, task, hidden_size=512,
                 embedding_matrix=None, dropout=0.5, init_type='xavier'):
        super(CustomLSTMModel, self).__init__()
        if task not in self.TARGET_INDEX:
            raise ValueError(f"task must be one of {list(self.TARGET_INDEX)}, got {task!r}")
        self.device = DEVICE
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.task = task
        self.target_idx = self.TARGET_INDEX[task]

        # Embedding layer: initialized with pre-trained GloVe matrix.
        # Weights are left trainable (requires_grad=True) to allow domain adaptation to WOS terminology.
        # clone() so fine-tuning never writes back into the shared matrix (on CPU, .to() would not copy it).
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix.clone())

        self.lstm_cell = CustomLSTMCell(embed_dim, hidden_size, init_type)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

        self.to(self.device)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x_emb = self.embedding(x)

        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Iterate over sequence steps with token masking for padding token 0
        for t in range(seq_len):
            x_t = x_emb[:, t, :]
            mask = (x[:, t] != 0).unsqueeze(1).float()  # 1.0 for valid words, 0.0 for <PAD>

            h_next, c_next = self.lstm_cell(x_t, (h_t, c_t))

            # Mask state update: freeze states if current token is <PAD>
            h_t = mask * h_next + (1.0 - mask) * h_t
            c_t = mask * c_next + (1.0 - mask) * c_t

        # Classification head on final valid unpadded hidden state
        out = self.dropout(h_t)
        logits = self.fc(out)
        return logits

    def fit(self, train_ldr, val_ldr, epochs=10, lr=0.001, l1_lambda=0.0, weight_decay=0.0):
        """
        Trains with Adam + gradient clipping, then restores the weights of the epoch with the best
        validation accuracy (the same criterion the hyperparameter sweeps use to rank configurations).
        """
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_acc, best_epoch, best_state = -1.0, 0, None
        print(f"\nTraining LSTM ({self.task}, output dim: {self.output_dim}) for {epochs} epochs...")

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for batch in train_ldr:
                X_batch = batch[0].to(self.device)
                target = batch[self.target_idx].to(self.device)

                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, target)

                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in self.parameters())
                    loss += l1_lambda * l1_norm

                loss.backward()
                # Clip gradients to max norm = 1.0 to prevent exploding gradients during BPTT
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            # Consolidated validation evaluation in single pass over val_ldr
            val_loss, val_acc = self.evaluate(val_ldr, criterion)

            history["train_loss"].append(total_loss / len(train_ldr))
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_ldr):.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch + 1
                best_state = copy.deepcopy(self.state_dict())

        if best_state is not None:
            self.load_state_dict(best_state)
            print(f"Restored best epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%)")

        return history

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            outputs = self(X)
            predictions = torch.argmax(outputs, dim=1)
        return predictions

    def evaluate(self, data_loader, criterion=None):
        """
        Evaluates loss and accuracy (%) in a single pass over data_loader to eliminate redundant passes.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                X_batch = batch[0].to(self.device)
                target = batch[self.target_idx].to(self.device)

                outputs = self(X_batch)
                loss = criterion(outputs, target)

                total_loss += loss.item() * target.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total
        avg_loss = total_loss / total
        return avg_loss, acc

    def evaluate_acc(self, data_loader):
        _, acc = self.evaluate(data_loader)
        return acc

    def evaluate_loss(self, data_loader, criterion):
        avg_loss, _ = self.evaluate(data_loader, criterion)
        return avg_loss

class BERTClassifier(nn.Module):
    """
    Fine-tuned Transformer Classifier wrapping HuggingFace `bert-base-uncased`.
    Uses the `pooler_output` (linear + Tanh transformation of [CLS] token) for classification.
    `output_attentions=True` is enabled for self-attention head visual inspection in Task 3.
    """
    def __init__(self, output_dim, model_name=BERT_MODEL_NAME, dropout=0.3):
        super().__init__()
        self.device = DEVICE
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        pooled_output = self.dropout(pooler_output)
        logits = self.fc(pooled_output)
        return logits

    def get_attention_maps(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device))
        return outputs.attentions

    def fit(self, train_ldr, val_ldr, lr=2e-5, epochs=3, patience=2,
            weight_decay=0.0, l1_lambda=0.0, checkpoint_name="best_bert_model.pth"):
        """
        Fine-tunes BERT using AdamW (decoupled weight decay) + linear warmup learning rate schedule.
        Saves best checkpoint based on validation loss and restores it upon completion.
        """
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_ldr) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Starting BERT fine-tuning (Checkpoint: {checkpoint_name})...")

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0
            correct_train = 0
            total_samples = 0
            start_time = time.time()

            for batch in train_ldr:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.zero_grad()
                logits = self.forward(b_input_ids, b_input_mask)
                loss = criterion(logits, b_labels)

                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in self.parameters())
                    loss += l1_lambda * l1_norm

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item() * b_input_ids.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_train += (preds == b_labels).sum().item()
                total_samples += b_input_ids.size(0)

            avg_train_loss = total_train_loss / total_samples
            train_acc = 100.0 * correct_train / total_samples

            val_loss, val_acc = self.evaluate(val_ldr, criterion)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), checkpoint_name)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        if os.path.exists(checkpoint_name):
            self.load_state_dict(torch.load(checkpoint_name, weights_only=True))
        return history

    def evaluate(self, val_ldr, criterion):
        """Returns (average loss, accuracy %) over val_ldr."""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_ldr:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                logits = self.forward(b_input_ids, b_input_mask)
                loss = criterion(logits, b_labels)

                total_loss += loss.item() * b_input_ids.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == b_labels).sum().item()
                total += b_input_ids.size(0)

        return total_loss / total, 100.0 * correct / total

"""## Task 3: Run experiments"""

def evaluate_lstm(train_ds, val_ds, vocab_size, output_dim, task, batch_size=BATCH_SIZE,
                  embedding_matrix=None, epochs=10, hidden_size=512, dropout=0.5,
                  l1_lambda=0.0, weight_decay=0.0, lr=0.001, init_type='xavier'):
    """Trains one LSTM configuration and returns its best validation accuracy (%)."""
    train_ldr = make_loader(train_ds, shuffle=True, batch_size=batch_size)
    val_ldr = make_loader(val_ds, batch_size=batch_size)

    model = CustomLSTMModel(
        vocab_size,
        embedding_matrix.shape[1],
        output_dim,
        task,
        hidden_size=hidden_size,
        dropout=dropout,
        embedding_matrix=embedding_matrix,
        init_type=init_type
    )

    history = model.fit(
        train_ldr, val_ldr,
        lr=lr,
        epochs=epochs,
        l1_lambda=l1_lambda,
        weight_decay=weight_decay
    )

    return max(history.get("val_acc", [0]))

def bert_encode(tokenizer, texts):
    """WordPiece-tokenizes texts (truncate to BERT_MAX_LEN, pad to the longest in the split)."""
    return tokenizer(texts, truncation=True, padding=True, max_length=BERT_MAX_LEN, return_tensors='pt')

def bert_dataset(encodings, labels):
    """Pairs pre-computed encodings with one label set (domain or sub-field) as (ids, mask, label)."""
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'],
                         torch.tensor(labels, dtype=torch.long))

def plot_combined_results(values, model_1_scores, model_2_scores, title, xlabel,
                          label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub", save_dir=FIGURES_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(values, model_1_scores, label=label_model_1, marker='o')
    ax.plot(values, model_2_scores, label=label_model_2, marker='x')

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    clean_title = title.replace(" ", "_").replace(":", "")
    filepath = os.path.join(save_dir, f"{clean_title}.png")

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {filepath}")

def plot_single_result(values, scores, title, xlabel, ylabel="Validation Accuracy (%)", save_dir=FIGURES_DIR):

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(values, scores, marker='o', label='LSTM')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    clean_title = title.replace(" ", "_").replace(":", "")
    filepath = os.path.join(save_dir, f"{clean_title}.png")

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to: {filepath}")

def report_ablation(name, values, scores, chosen):
    """Prints an ablation's winner next to the value the final model actually uses."""
    best = values[int(np.argmax(scores))]
    note = "" if best == chosen else " (differs from the sweep winner; single-seed gaps are within run-to-run noise)"
    print(f"[Ablation] {name}: sweep best = {best}, final model uses = {chosen}{note}")

def run_lstm_sweeps(train_ds, val_ds, vocab, embedding_matrix, num_domains, num_subfields):
    """
    Greedy sequential search over dropout -> learning rate -> hidden size for both tasks,
    followed by domain-task ablations over initialization, L1 strength and GloVe dimension.
    Returns the selected dropout / learning rate / hidden size per task.
    """
    vocab_size = len(vocab)
    domain_kwargs = dict(output_dim=num_domains, task="domain", epochs=LSTM_SWEEP_EPOCHS_DOMAIN)
    sub_kwargs = dict(output_dim=num_subfields, task="sub", epochs=LSTM_SWEEP_EPOCHS_SUB)

    # 1) Test Dropout for LSTM
    dropout_list = [0.1, 0.3, 0.5]
    lstm_res_domain = []
    lstm_res_sub = []

    for d in dropout_list:
        print(f"Testing dropout: {d}")
        lstm_res_domain.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=embedding_matrix,
                                             dropout=d, **domain_kwargs))
        lstm_res_sub.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=embedding_matrix,
                                          dropout=d, **sub_kwargs))

    plot_combined_results(dropout_list, lstm_res_domain, lstm_res_sub, "Effect of Dropout", "Dropout Rate",
                          label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

    best_dropout_domain = dropout_list[np.argmax(lstm_res_domain)]
    best_dropout_sub = dropout_list[np.argmax(lstm_res_sub)]

    # 2) Test Learning Rate for LSTM
    lr_list = [1e-2, 1e-3, 1e-4]
    lstm_res_domain_lr = []
    lstm_res_sub_lr = []

    for lr in lr_list:
        print(f"Testing learning rate: {lr}")
        lstm_res_domain_lr.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=embedding_matrix,
                                                dropout=best_dropout_domain, lr=lr, **domain_kwargs))
        lstm_res_sub_lr.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=embedding_matrix,
                                             dropout=best_dropout_sub, lr=lr, **sub_kwargs))

    plot_combined_results(lr_list, lstm_res_domain_lr, lstm_res_sub_lr, "Effect of Learning Rate", "Learning Rate",
                          label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

    best_lr_domain = lr_list[np.argmax(lstm_res_domain_lr)]
    best_lr_sub = lr_list[np.argmax(lstm_res_sub_lr)]

    # 3) Test Hidden Size for LSTM
    hidden_size_list = [128, 256, 512]
    lstm_res_domain_hidden = []
    lstm_res_sub_hidden = []

    for hs in hidden_size_list:
        print(f"Testing hidden size: {hs}")
        lstm_res_domain_hidden.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=embedding_matrix,
                                                    dropout=best_dropout_domain, lr=best_lr_domain,
                                                    hidden_size=hs, **domain_kwargs))
        lstm_res_sub_hidden.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=embedding_matrix,
                                                 dropout=best_dropout_sub, lr=best_lr_sub,
                                                 hidden_size=hs, **sub_kwargs))

    plot_combined_results(hidden_size_list, lstm_res_domain_hidden, lstm_res_sub_hidden, "Effect of Hidden Size", "Hidden Size",
                          label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

    best_hs_domain = hidden_size_list[np.argmax(lstm_res_domain_hidden)]
    best_hs_sub = hidden_size_list[np.argmax(lstm_res_sub_hidden)]

    tuned_domain = dict(dropout=best_dropout_domain, lr=best_lr_domain, hidden_size=best_hs_domain)

    # 4) Ablation: Initialization Type (Domain Task Only)
    init_types = ['xavier', 'random', 'zero']
    init_results = []

    for init_type in init_types:
        print(f"Testing initialization type: {init_type}")
        init_results.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=embedding_matrix,
                                          init_type=init_type, **tuned_domain, **domain_kwargs))

    plot_single_result(init_types, init_results, "Effect of Initialization", "Init Type")
    report_ablation("Initialization", init_types, init_results, FINAL_INIT_TYPE)

    # 5) Ablation: L1 Regularization (Domain Task Only)
    l1_values = [0.0, 1e-5, 1e-3]
    l1_results = []

    for l1_val in l1_values:
        print(f"Testing L1 lambda: {l1_val}")
        l1_results.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=embedding_matrix,
                                        l1_lambda=l1_val, **tuned_domain, **domain_kwargs))

    plot_single_result([str(v) for v in l1_values], l1_results, "Effect of L1 Regularization", "L1 Lambda")
    report_ablation("L1 lambda", l1_values, l1_results, FINAL_L1_LAMBDA)

    # 6) Ablation: GloVe Embedding Dimension (50d vs 100d vs 200d vs 300d)
    print("GloVe dimension comparison...")
    glove_dims = [50, 100, 200, 300]
    dim_results = []

    for dim in glove_dims:
        path = os.path.join(GLOVE_DIR, f"glove.6B.{dim}d.txt")
        print(f"\nTesting GloVe dimension: {dim}d")
        current_matrix = load_glove_matrix(path, vocab, embed_dim=dim)
        dim_results.append(evaluate_lstm(train_ds, val_ds, vocab_size, embedding_matrix=current_matrix,
                                         **tuned_domain, **domain_kwargs))

    plot_single_result([str(d) for d in glove_dims], dim_results, "Effect of Embedding Dimension", "GloVe Dimension")
    report_ablation("GloVe dimension", glove_dims, dim_results, EMBED_DIM)

    return {
        "domain": tuned_domain,
        "sub": dict(dropout=best_dropout_sub, lr=best_lr_sub, hidden_size=best_hs_sub),
    }

"""## Attention Analysis & Visualization"""

def get_sample_data(model, data_loader):
    """
    Identifies one correctly predicted and one incorrectly predicted document
    sample directly from the dataset tensors for attention visual analysis.
    Either entry is None if no such sample exists (e.g. a perfectly accurate model).
    """
    model.eval()
    correct_data = None
    incorrect_data = None

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids, b_input_mask, b_labels = [b.to(model.device) for b in batch]

            logits = model(b_input_ids, b_input_mask)
            preds = torch.argmax(logits, dim=1)

            correct_mask = preds == b_labels
            incorrect_mask = preds != b_labels

            if correct_data is None and correct_mask.any():
                idx = torch.where(correct_mask)[0][0]
                correct_data = (b_input_ids[idx].unsqueeze(0), b_input_mask[idx].unsqueeze(0))

            if incorrect_data is None and incorrect_mask.any():
                idx = torch.where(incorrect_mask)[0][0]
                incorrect_data = (b_input_ids[idx].unsqueeze(0), b_input_mask[idx].unsqueeze(0))

            if correct_data is not None and incorrect_data is not None:
                break

    return correct_data, incorrect_data

def visualize_attention(model, tokenizer, input_ids, attention_mask, title_prefix, device,
                        layer_idx=11, head_idx=0, save_dir=FIGURES_DIR):
    """
    Extracts and plots self-attention heatmap matrix for Layer `layer_idx+1` and Head `head_idx+1`.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    attentions = model.get_attention_maps(input_ids, attention_mask)
    attention_matrix = attentions[layer_idx][0, head_idx, :, :].cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    seq_len = attention_mask.sum().item()
    attention_matrix = attention_matrix[:seq_len, :seq_len]
    tokens = tokens[:seq_len]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis', ax=ax)

    title = f"{title_prefix} - Layer {layer_idx+1}, Head {head_idx+1}"
    ax.set_title(title)
    ax.set_xlabel("Key (Attended To)")
    ax.set_ylabel("Query (Attending From)")
    plt.xticks(rotation=90)

    clean_title = title.replace(" ", "_").replace(":", "").replace(",", "")
    filepath = os.path.join(save_dir, f"{clean_title}.png")

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved attention map to: {filepath}")

def visualize_token_importance(model, tokenizer, input_ids, attention_mask, title_prefix, device,
                               layer_idx=11, head_idx=0, save_dir=FIGURES_DIR):

    """
    Plots horizontal bar chart of the top 15 non-special tokens receiving maximum attention
    weight from the classification [CLS] token.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    attentions = model.get_attention_maps(input_ids, attention_mask)
    cls_attention = attentions[layer_idx][0, head_idx, 0, :].cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    valid_tokens = []
    valid_scores = []
    for token, score in zip(tokens, cls_attention):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            valid_tokens.append(token)
            valid_scores.append(score)

    sorted_indices = np.argsort(valid_scores)[::-1]
    top_n = 15

    top_tokens = [valid_tokens[i] for i in sorted_indices[:top_n]]
    top_scores = [valid_scores[i] for i in sorted_indices[:top_n]]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_tokens))
    ax.barh(y_pos, top_scores, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_tokens)
    ax.invert_yaxis()

    title = f"{title_prefix} Top Tokens - Layer {layer_idx+1} Head {head_idx+1}"
    ax.set_xlabel('Attention Score')
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    clean_title = title.replace(" ", "_").replace(":", "").replace("-", "_")
    filepath = os.path.join(save_dir, f"{clean_title}.png")

    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved token importance plot to: {filepath}")

def plot_model_comparison(test_acc_domain_lstm, test_acc_domain_bert,
                          test_acc_sub_lstm, test_acc_sub_bert,
                          num_domains, num_subfields,
                          save_path=os.path.join(FIGURES_DIR, "model_comparison.png")):

    tasks = [f'Domain ({num_domains} Classes)', f'Sub-field ({num_subfields} Classes)']
    lstm_scores = [test_acc_domain_lstm, test_acc_sub_lstm]
    bert_scores = [test_acc_domain_bert, test_acc_sub_bert]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    rects1 = ax.bar(x - width/2, lstm_scores, width, label='Custom LSTM (GloVe)', color='skyblue')
    rects2 = ax.bar(x + width/2, bert_scores, width, label='BERT Classifier', color='salmon')

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Comparison plot saved to {save_path}")

def main():
    print(f"Using device: {DEVICE}")

    # Initialize and prepare datasets
    print("Loading data...")
    train_ds, val_ds, test_ds, embedding_matrix, vocab = prepare_data()
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds)

    # BERT gets the original (uncleaned) text: its own uncased WordPiece tokenizer handles casing and punctuation
    bert_texts, y_sub_raw, y_domain_raw = load_data(DATA_DIR, clean=False)

    # Class counts come from the label files, so an unexpected label id can never index past the output layer
    num_domains = max(y_domain_raw) + 1
    num_subfields = max(y_sub_raw) + 1
    print(f"Classes: {num_domains} domains, {num_subfields} sub-fields "
          f"({len(set(y_sub_raw))} sub-field ids present in the data).")

    train_indices = train_ds.indices
    val_indices = val_ds.indices
    test_indices = test_ds.indices

    # Pre-tokenize BERT text data ONCE per split; domain and sub-field datasets share the encodings
    print("Pre-tokenizing texts for BERT...")
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_train_enc = bert_encode(bert_tokenizer, [bert_texts[i] for i in train_indices])
    bert_val_enc = bert_encode(bert_tokenizer, [bert_texts[i] for i in val_indices])
    bert_test_enc = bert_encode(bert_tokenizer, [bert_texts[i] for i in test_indices])

    def labels_for(labels, indices):
        return [labels[i] for i in indices]

    # ---------------------------------------------------------
    # Hyperparameter Search Sweeps (Greedy Sequential Search)
    # ---------------------------------------------------------
    best = run_lstm_sweeps(train_ds, val_ds, vocab, embedding_matrix, num_domains, num_subfields)
    print(f"Selected LSTM config -> domain: {best['domain']} | sub-field: {best['sub']}")

    # Free memory before final model training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Final Classification Experiments (4 Runs)
    # ---------------------------------------------------------

    # 1) Custom LSTM (Domain Classification)
    print(f"\n--- EXP 1/4: Custom LSTM (Domain Classification, {num_domains} classes) ---")
    lstm_domain = CustomLSTMModel(
        vocab_size=len(vocab), embed_dim=EMBED_DIM, output_dim=num_domains, task="domain",
        hidden_size=best["domain"]["hidden_size"], embedding_matrix=embedding_matrix,
        dropout=best["domain"]["dropout"], init_type=FINAL_INIT_TYPE
    )
    lstm_domain.fit(train_loader, val_loader, epochs=LSTM_FINAL_EPOCHS_DOMAIN,
                    lr=best["domain"]["lr"], l1_lambda=FINAL_L1_LAMBDA)
    test_acc_domain_lstm = lstm_domain.evaluate_acc(test_loader)
    print(f"Final LSTM Domain Test Accuracy: {test_acc_domain_lstm:.2f}%")

    # 2) BERT Fine-Tuning (Domain Classification)
    print(f"\n--- EXP 2/4: BERT Fine-Tuning (Domain Classification, {num_domains} classes) ---")
    bert_domain_model = BERTClassifier(output_dim=num_domains)

    bert_domain_train_ldr = make_loader(bert_dataset(bert_train_enc, labels_for(y_domain_raw, train_indices)), shuffle=True)
    bert_domain_val_ldr = make_loader(bert_dataset(bert_val_enc, labels_for(y_domain_raw, val_indices)))
    bert_domain_test_ldr = make_loader(bert_dataset(bert_test_enc, labels_for(y_domain_raw, test_indices)))

    bert_domain_model.fit(bert_domain_train_ldr, bert_domain_val_ldr, epochs=BERT_EPOCHS,
                          checkpoint_name="best_bert_domain.pth")
    _, test_acc_domain_bert = bert_domain_model.evaluate(bert_domain_test_ldr, nn.CrossEntropyLoss())
    print(f"Final BERT Domain Test Accuracy: {test_acc_domain_bert:.2f}%")

    # 3) Custom LSTM (Sub-field Classification)
    print(f"\n--- EXP 3/4: Custom LSTM (Sub-field Classification, {num_subfields} classes) ---")
    lstm_sub = CustomLSTMModel(
        vocab_size=len(vocab), embed_dim=EMBED_DIM, output_dim=num_subfields, task="sub",
        hidden_size=best["sub"]["hidden_size"], embedding_matrix=embedding_matrix,
        dropout=best["sub"]["dropout"], init_type=FINAL_INIT_TYPE
    )
    lstm_sub.fit(train_loader, val_loader, epochs=LSTM_FINAL_EPOCHS_SUB,
                 lr=best["sub"]["lr"], l1_lambda=FINAL_L1_LAMBDA)
    test_acc_sub_lstm = lstm_sub.evaluate_acc(test_loader)
    print(f"Final LSTM Sub-field Test Accuracy: {test_acc_sub_lstm:.2f}%")

    # 4) BERT Fine-Tuning (Sub-field Classification)
    print(f"\n--- EXP 4/4: BERT Fine-Tuning (Sub-field Classification, {num_subfields} classes) ---")
    bert_sub_model = BERTClassifier(output_dim=num_subfields)

    bert_sub_train_ldr = make_loader(bert_dataset(bert_train_enc, labels_for(y_sub_raw, train_indices)), shuffle=True)
    bert_sub_val_ldr = make_loader(bert_dataset(bert_val_enc, labels_for(y_sub_raw, val_indices)))
    bert_sub_test_ldr = make_loader(bert_dataset(bert_test_enc, labels_for(y_sub_raw, test_indices)))

    bert_sub_model.fit(bert_sub_train_ldr, bert_sub_val_ldr, epochs=BERT_EPOCHS,
                       checkpoint_name="best_bert_subfield.pth")
    _, test_acc_sub_bert = bert_sub_model.evaluate(bert_sub_test_ldr, nn.CrossEntropyLoss())
    print(f"Final BERT Sub-field Test Accuracy: {test_acc_sub_bert:.2f}%")

    # ---------------------------------------------------------
    # Attention Analysis & Visualization (BERT Domain model, test set)
    # ---------------------------------------------------------
    correct_data, incorrect_data = get_sample_data(bert_domain_model, bert_domain_test_ldr)
    last_layer = bert_domain_model.bert.config.num_hidden_layers - 1  # Layer 12 for bert-base

    for prefix, sample in (("Correct", correct_data), ("Incorrect", incorrect_data)):
        if sample is None:
            print(f"No {prefix.lower()} test prediction found; skipping its attention plots.")
            continue
        visualize_attention(bert_domain_model, bert_tokenizer, sample[0], sample[1], prefix, DEVICE,
                            layer_idx=last_layer, head_idx=0)
        visualize_token_importance(bert_domain_model, bert_tokenizer, sample[0], sample[1], prefix, DEVICE,
                                   layer_idx=last_layer, head_idx=0)

    # Summary Output Table
    domain_task = f"Domain ({num_domains} Classes)"
    sub_task = f"Sub-field ({num_subfields} Classes)"
    print("\n--- RESULTS SUMMARY TABLE ---")
    print("| Model | Task | Test Accuracy | Winner? |")
    print("|---|---|---|---|")
    print(f"| Custom LSTM (GloVe) | {domain_task} | {test_acc_domain_lstm:.2f}% | {'<--' if test_acc_domain_lstm > test_acc_domain_bert else ''} |")
    print(f"| BERT Classifier | {domain_task} | {test_acc_domain_bert:.2f}% | {'<--' if test_acc_domain_bert > test_acc_domain_lstm else ''} |")
    print(f"| Custom LSTM (GloVe) | {sub_task} | {test_acc_sub_lstm:.2f}% | {'<--' if test_acc_sub_lstm > test_acc_sub_bert else ''} |")
    print(f"| BERT Classifier | {sub_task} | {test_acc_sub_bert:.2f}% | {'<--' if test_acc_sub_bert > test_acc_sub_lstm else ''} |")

    plot_model_comparison(test_acc_domain_lstm, test_acc_domain_bert, test_acc_sub_lstm, test_acc_sub_bert,
                          num_domains, num_subfields)

# Guard required on Windows/macOS: DataLoader workers re-import this module, which must not re-run the pipeline
if __name__ == "__main__":
    main()
