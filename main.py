import os
import re
import time
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device

DEVICE = getDevice()

# DataLoader hardware optimization constants
NUM_WORKERS = min(4, os.cpu_count() if os.cpu_count() else 0)
PIN_MEMORY = True if DEVICE.type == "cuda" else False

"""## Task 1: Acquire and Pre-process the Web of Science Dataset"""

DATA_DIR = "WebOfScienceDataset/WOS11967"
GLOVE_PATH = "glove.6B/glove.6B.300d.txt"

MAX_SEQ_LEN = 200  # Dynamically set via 95th percentile of training token length
EMBED_DIM = 300    # 300-dimensional GloVe pre-trained vectors
MAX_VOCAB = 10000  # Maximum vocabulary size (top 10,000 frequent terms)
BATCH_SIZE = 32

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

def load_data(data_dir):
    """
    Loads raw text abstracts (X), 33-class sub-field targets (Y), and
    7-class parent domain targets (YL1) from the dataset directory.
    """
    with open(os.path.join(data_dir, 'X.txt'), 'r', encoding='utf-8') as f:
        texts = [clean_text(line) for line in f.readlines()]

    with open(os.path.join(data_dir, 'Y.txt'), 'r', encoding='utf-8') as f:
        y_sub = [int(line.strip()) for line in f.readlines()]

    with open(os.path.join(data_dir, 'YL1.txt'), 'r', encoding='utf-8') as f:
        y_domain = [int(line.strip()) for line in f.readlines()]

    return texts, y_sub, y_domain

def get_dynamic_max_len(texts, percentile=95):
    """
    Calculates the sequence length covering `percentile`% of all input documents.
    This dynamically balances context retention against wasteful zero-padding memory overhead.
    """
    lengths = [len(t.split()) for t in texts]
    limit = int(np.percentile(lengths, percentile))
    print(f"95th percentile length is {limit}. Mean is {int(np.mean(lengths))}.")
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

def prepare_data():
    """
    Prepares dataset, constructs vocabulary, loads GloVe embedding matrix,
    and performs 64/16/20 Train/Validation/Test random split.
    """
    texts, y_sub, y_domain = load_data(DATA_DIR)

    dynamic_max_len = get_dynamic_max_len(texts)
    vocab = build_vocab(texts)

    X_tensor = encode_texts(texts, vocab, max_len=dynamic_max_len)
    Y_sub_tensor = torch.tensor(y_sub, dtype=torch.long)
    Y_domain_tensor = torch.tensor(y_domain, dtype=torch.long)

    embedding_weights = load_glove_matrix(GLOVE_PATH, vocab)

    dataset = TensorDataset(X_tensor, Y_domain_tensor, Y_sub_tensor)

    # Train/Val/Test Split (64% Train / 16% Val / 20% Test)
    test_size = int(0.2 * len(dataset))
    remaining_size = len(dataset) - test_size
    val_size = int(0.2 * remaining_size)
    train_size = remaining_size - val_size

    # Reproducible generator seed for random_split
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_ldr = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_ldr = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_ldr = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_ldr, val_ldr, test_ldr, embedding_weights, vocab, train_dataset, val_dataset, test_dataset

"""## Task 2: Implement LSTM and BERT models"""

class CustomLSTMCell(nn.Module):
    """
    From-scratch LSTM Cell implementing gating mathematics manually:
    - i_t = σ(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)  [Input gate]
    - f_t = σ(W_fh @ x_t + b_fh + W_fh @ h_{t-1} + b_fh)  [Forget gate]
    - g_t = tanh(W_gh @ x_t + b_gh + W_gh @ h_{t-1} + b_gh) [Candidate cell state]
    - o_t = σ(W_oh @ x_t + b_oh + W_oh @ h_{t-1} + b_oh)  [Output gate]
    - c_t = f_t * c_{t-1} + i_t * g_t                      [Cell state update]
    - h_t = o_t * tanh(c_t)                                [Hidden state output]
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
    """
    def __init__(self, vocab_size, embed_dim, output_dim, hidden_size=512,
                 embedding_matrix=None, dropout=0.5, init_type='xavier'):
        super(CustomLSTMModel, self).__init__()
        self.device = DEVICE
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # Embedding layer: initialized with pre-trained GloVe matrix.
        # Weights are left trainable (requires_grad=True) to allow domain adaptation to WOS terminology.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)

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
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        print(f"\nTraining LSTM (output dim: {self.output_dim}) for {epochs} epochs...")

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for X_batch, Y_domain, Y_sub in train_ldr:
                X_batch = X_batch.to(self.device)
                target = Y_domain.to(self.device) if self.output_dim == 7 else Y_sub.to(self.device)

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

            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_ldr):.4f} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

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
        Evaluates loss and accuracy in a single pass over data_loader to eliminate redundant passes.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, Y_domain, Y_sub in data_loader:
                X_batch = X_batch.to(self.device)
                target = Y_domain.to(self.device) if self.output_dim == 7 else Y_sub.to(self.device)

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
    def __init__(self, output_dim, model_name='bert-base-uncased', dropout=0.3):
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
            train_acc = correct_train / total_samples

            val_loss, val_acc = self.evaluate(val_ldr, criterion)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s] | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

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

        return total_loss / total, correct / total

"""## Task 3: Run experiments"""

def evaluate_lstm(train_ds, val_ds, vocab_size, output_dim, batch_size=BATCH_SIZE,
                  embedding_matrix=None, epochs=10, hidden_size=512, dropout=0.5,
                  l1_lambda=0.0, weight_decay=0.0, lr=0.001, init_type='xavier'):

    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = CustomLSTMModel(
        vocab_size,
        embedding_matrix.shape[1],
        output_dim,
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

def evaluate_bert(dropout, l1_lambda, weight_decay, batch_size, lr,
                  train_ds, val_ds, output_dim):

    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_ldr = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = BERTClassifier(output_dim, dropout=dropout).to(DEVICE)

    history = model.fit(
        train_ldr, val_ldr,
        lr=lr,
        epochs=2,
        patience=1,
        weight_decay=weight_decay,
        l1_lambda=l1_lambda,
        checkpoint_name="temp_search_bert.pth"
    )

    return max(history.get("val_acc", [0]))

def get_bert_test_data(all_texts, all_labels, test_indices, max_length=256):
    """
    Prepares a TensorDataset and DataLoader for the BERT test set.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_texts = [all_texts[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    test_dataset = TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        torch.tensor(test_labels)
    )
    test_ldr = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return test_ldr, test_texts, test_labels, tokenizer

def plot_combined_results(values, model_1_scores, model_2_scores, title, xlabel,
                          label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub", save_dir="figures"):
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

def plot_single_result(values, scores, title, xlabel, ylabel="Validation Accuracy (%)", save_dir="figures"):

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

# Initialize and prepare datasets
print("Loading data...")
train_loader, val_loader, test_loader, embedding_matrix, vocab, train_ds, val_ds, test_ds = prepare_data()

all_texts, y_sub_raw, y_domain_raw = load_data(DATA_DIR)

vocab_size = len(vocab)
output_dim_domain = 7
output_dim_sub = 33

train_indices = train_ds.indices
val_indices = val_ds.indices
test_indices = test_ds.indices

bert_train_labels = [y_domain_raw[i] for i in train_indices]
bert_val_labels = [y_domain_raw[i] for i in val_indices]

bert_sub_train_labels = [y_sub_raw[i] for i in train_indices]
bert_sub_val_labels = [y_sub_raw[i] for i in val_indices]

bert_train_texts = [all_texts[i] for i in train_indices]
bert_val_texts = [all_texts[i] for i in val_indices]

# Pre-tokenize BERT text data ONCE to eliminate redundant tokenization loops
print("Pre-tokenizing texts for BERT...")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_train_enc = bert_tokenizer(bert_train_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
bert_val_enc = bert_tokenizer(bert_val_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')

# ---------------------------------------------------------
# Hyperparameter Search Sweeps (Greedy Sequential Search)
# ---------------------------------------------------------

# 1) Test Dropout for LSTM
dropout_list = [0.1, 0.3, 0.5]
lstm_res_domain = []
lstm_res_sub = []

for d in dropout_list:
    print(f"Testing dropout: {d}")
    l_acc_domain = evaluate_lstm(train_ds, val_ds, vocab_size, output_dim_domain, embedding_matrix=embedding_matrix, dropout=d)
    lstm_res_domain.append(l_acc_domain)

    l_acc_sub = evaluate_lstm(train_ds, val_ds, vocab_size, output_dim_sub, embedding_matrix=embedding_matrix, dropout=d, epochs=20)
    lstm_res_sub.append(l_acc_sub)

plot_combined_results(dropout_list, lstm_res_domain, lstm_res_sub, "Effect of Dropout", "Dropout Rate",
                      label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

BEST_DROPOUT_LSTM_DOMAIN = dropout_list[np.argmax(lstm_res_domain)]
BEST_DROPOUT_LSTM_SUB = dropout_list[np.argmax(lstm_res_sub)]

# 2) Test Learning Rate for LSTM
lr_list = [1e-2, 1e-3, 1e-4]
lstm_res_domain_lr = []
lstm_res_sub_lr = []

for lr in lr_list:
    print(f"Testing learning rate: {lr}")
    l_acc_domain = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=lr
    )
    lstm_res_domain_lr.append(l_acc_domain)

    l_acc_sub = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_sub,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_SUB,
        lr=lr,
        epochs=20
    )
    lstm_res_sub_lr.append(l_acc_sub)

plot_combined_results(lr_list, lstm_res_domain_lr, lstm_res_sub_lr, "Effect of Learning Rate", "Learning Rate",
                      label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

BEST_LR_LSTM_DOMAIN = lr_list[np.argmax(lstm_res_domain_lr)]
BEST_LR_LSTM_SUB = lr_list[np.argmax(lstm_res_sub_lr)]

# 3) Test Hidden Size for LSTM
hidden_size_list = [128, 256, 512]
lstm_res_domain_hidden = []
lstm_res_sub_hidden = []

for hs in hidden_size_list:
    print(f"Testing hidden size: {hs}")
    l_acc_domain = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=BEST_LR_LSTM_DOMAIN,
        hidden_size=hs
    )
    lstm_res_domain_hidden.append(l_acc_domain)

    l_acc_sub = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_sub,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_SUB,
        lr=BEST_LR_LSTM_SUB,
        hidden_size=hs,
        epochs=20
    )
    lstm_res_sub_hidden.append(l_acc_sub)

plot_combined_results(hidden_size_list, lstm_res_domain_hidden, lstm_res_sub_hidden, "Effect of Hidden Size", "Hidden Size",
                      label_model_1="LSTM Val Acc Domain", label_model_2="LSTM Val Acc Sub")

BEST_HS_LSTM_DOMAIN = hidden_size_list[np.argmax(lstm_res_domain_hidden)]
BEST_HS_LSTM_SUB = hidden_size_list[np.argmax(lstm_res_sub_hidden)]

# 4) Test Initialization Type for LSTM (Fixed bug: pass init_type=hs_init)
init_types = ['xavier', 'random', 'zero']
init_results = []

for hs_init in init_types:
    print(f"Testing initialization type: {hs_init}")
    l_acc_domain = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=BEST_LR_LSTM_DOMAIN,
        hidden_size=BEST_HS_LSTM_DOMAIN,
        init_type=hs_init
    )
    init_results.append(l_acc_domain)

plot_single_result(init_types, init_results, "Effect of Initialization", "Init Type")

# 5) Test L1 Regularization for LSTM (Domain Task Only)
l1_values = [0.0, 1e-5, 1e-3]
l1_results = []

for l1_val in l1_values:
    print(f"Testing L1 lambda: {l1_val}")
    l_acc_domain = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=embedding_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=BEST_LR_LSTM_DOMAIN,
        hidden_size=BEST_HS_LSTM_DOMAIN,
        l1_lambda=l1_val
    )
    l1_results.append(l_acc_domain)

l1_labels = [str(v) for v in l1_values]
plot_single_result(l1_labels, l1_results, "Effect of L1 Regularization", "L1 Lambda")

# 6) GloVe Embedding Dimension Comparison (50d vs 100d vs 200d vs 300d)
print("GloVe dimension comparison...")
GLOVE_DIR = "glove.6B"
glove_dims = [50, 100, 200, 300]
dim_results = []

for dim in glove_dims:
    filename = f"glove.6B.{dim}d.txt"
    path = os.path.join(GLOVE_DIR, filename)

    print(f"\nTesting GloVe dimension: {dim}d")
    current_matrix = load_glove_matrix(path, vocab, embed_dim=dim)

    acc = evaluate_lstm(
        train_ds, val_ds, vocab_size, output_dim_domain,
        embedding_matrix=current_matrix,
        dropout=BEST_DROPOUT_LSTM_DOMAIN,
        lr=BEST_LR_LSTM_DOMAIN,
        hidden_size=BEST_HS_LSTM_DOMAIN,
        epochs=10
    )
    dim_results.append(acc)

dim_labels = [str(d) for d in glove_dims]
plot_single_result(dim_labels, dim_results, "Effect of Embedding Dimension", "GloVe Dimension")

# Free memory before final model training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# Final Classification Experiments (4 Runs)
# ---------------------------------------------------------

# 1) Custom LSTM (Domain Classification - 7 classes)
print("\n--- EXP 1/4: Custom LSTM (Domain Classification) ---")
lstm_domain = CustomLSTMModel(
    vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_size=BEST_HS_LSTM_DOMAIN,
    output_dim=output_dim_domain, embedding_matrix=embedding_matrix,
    dropout=BEST_DROPOUT_LSTM_DOMAIN
)
lstm_domain.fit(train_loader, val_loader, epochs=10)
test_acc_domain_lstm = lstm_domain.evaluate_acc(test_loader)
print(f"Final LSTM Domain Test Accuracy: {test_acc_domain_lstm:.2f}%")

# 2) BERT Fine-Tuning (Domain Classification - 7 classes)
print("\n--- EXP 2/4: BERT Fine-Tuning (Domain Classification) ---")
bert_domain_model = BERTClassifier(output_dim=output_dim_domain)

bert_domain_train_ldr = DataLoader(
    TensorDataset(bert_train_enc['input_ids'], bert_train_enc['attention_mask'], torch.tensor(bert_train_labels)),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
bert_domain_val_ldr = DataLoader(
    TensorDataset(bert_val_enc['input_ids'], bert_val_enc['attention_mask'], torch.tensor(bert_val_labels)),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

bert_domain_model.fit(bert_domain_train_ldr, bert_domain_val_ldr, epochs=3, checkpoint_name="best_bert_domain.pth")

bert_test_loader, _, _, _ = get_bert_test_data(all_texts, y_domain_raw, test_indices)
_, bert_test_acc = bert_domain_model.evaluate(bert_test_loader, nn.CrossEntropyLoss())
test_acc_domain_bert = bert_test_acc * 100
print(f"Final BERT Domain Test Accuracy: {test_acc_domain_bert:.2f}%")

# 3) Custom LSTM (Sub-field Classification - 33 classes)
print("\n--- EXP 3/4: Custom LSTM (Sub-field Classification) ---")
lstm_sub = CustomLSTMModel(
    vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_size=BEST_HS_LSTM_SUB,
    output_dim=output_dim_sub, embedding_matrix=embedding_matrix,
    dropout=BEST_DROPOUT_LSTM_SUB
)
lstm_sub.fit(train_loader, val_loader, epochs=25)
test_acc_sub_lstm = lstm_sub.evaluate_acc(test_loader)
print(f"Final LSTM Sub-field Test Accuracy: {test_acc_sub_lstm:.2f}%")

# 4) BERT Fine-Tuning (Sub-field Classification - 33 classes)
print("\n--- EXP 4/4: BERT Fine-Tuning (Sub-field Classification) ---")
bert_sub_model = BERTClassifier(output_dim=output_dim_sub)

bert_sub_train_ldr = DataLoader(
    TensorDataset(bert_train_enc['input_ids'], bert_train_enc['attention_mask'], torch.tensor(bert_sub_train_labels)),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
bert_sub_val_ldr = DataLoader(
    TensorDataset(bert_val_enc['input_ids'], bert_val_enc['attention_mask'], torch.tensor(bert_sub_val_labels)),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

bert_sub_model.fit(bert_sub_train_ldr, bert_sub_val_ldr, epochs=3, checkpoint_name="best_bert_subfield.pth")

bert_sub_test_loader, _, _, _ = get_bert_test_data(all_texts, y_sub_raw, test_indices)
_, bert_sub_test_acc = bert_sub_model.evaluate(bert_sub_test_loader, nn.CrossEntropyLoss())
test_acc_sub_bert = bert_sub_test_acc * 100
print(f"Final BERT Sub-field Test Accuracy: {test_acc_sub_bert:.2f}%")

# ---------------------------------------------------------
# Attention Analysis & Visualization
# ---------------------------------------------------------

def get_sample_data(model, data_loader):
    """
    Identifies one correctly predicted and one incorrectly predicted document
    sample directly from the dataset tensors for attention visual analysis.
    """
    model.eval()
    correct_data = None
    incorrect_data = None

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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

    return correct_data, incorrect_data, tokenizer

def visualize_attention(model, tokenizer, input_ids, attention_mask, title_prefix, device,
                        layer_idx=11, head_idx=0, save_dir="figures"):
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
                               layer_idx=11, head_idx=0, save_dir="figures"):

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

# Execute BERT Domain Attention Analysis
bert_test_loader_domain, _, _, _ = get_bert_test_data(all_texts, y_domain_raw, test_indices)
correct_data, incorrect_data, tokenizer = get_sample_data(bert_domain_model, bert_test_loader_domain)

visualize_attention(bert_domain_model, tokenizer, correct_data[0], correct_data[1], "Correct", DEVICE)
visualize_attention(bert_domain_model, tokenizer, incorrect_data[0], incorrect_data[1], "Incorrect", DEVICE)

visualize_token_importance(bert_domain_model, tokenizer, correct_data[0], correct_data[1], "Correct", DEVICE, layer_idx=11, head_idx=0)
visualize_token_importance(bert_domain_model, tokenizer, incorrect_data[0], incorrect_data[1], "Incorrect", DEVICE, layer_idx=11, head_idx=0)

# Summary Output Table
print("\n--- RESULTS SUMMARY TABLE ---")
print("| Model | Task | Test Accuracy | Winner? |")
print("|---|---|---|---|")
print(f"| Custom LSTM (GloVe) | Domain (7 Classes) | {test_acc_domain_lstm:.2f}% | {'<--' if test_acc_domain_lstm > test_acc_domain_bert else ''} |")
print(f"| BERT Classifier | Domain (7 Classes) | {test_acc_domain_bert:.2f}% | {'<--' if test_acc_domain_bert > test_acc_domain_lstm else ''} |")
print(f"| Custom LSTM (GloVe) | Sub-field (33 Classes) | {test_acc_sub_lstm:.2f}% | {'<--' if test_acc_sub_lstm > test_acc_sub_bert else ''} |")
print(f"| BERT Classifier | Sub-field (33 Classes) | {test_acc_sub_bert:.2f}% | {'<--' if test_acc_sub_bert > test_acc_sub_lstm else ''} |")

def plot_model_comparison(test_acc_domain_lstm, test_acc_domain_bert,
                          test_acc_sub_lstm, test_acc_sub_bert,
                          save_path="figures/model_comparison.png"):


    tasks = ['Domain (7 Classes)', 'Sub-field (33 Classes)']
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

plot_model_comparison(test_acc_domain_lstm, test_acc_domain_bert, test_acc_sub_lstm, test_acc_sub_bert)
