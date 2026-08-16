import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from transformer.model import Embedding, PositionalEncoding, FeedForward, ResidualConnections, ProjectionLayer
from transformer.tokenizer import Tokenizer
from transformer.config import transformer_configurations


def load_data():
    dataset = load_dataset("roneneldan/TinyStories", split="train[:20000]")
    return dataset["text"]


class TextDataset(Dataset):

    def __init__(self, sentences, tok, max_len):
        super().__init__()
        self.sentences = sentences
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tok.encode_sentence(self.sentences[idx], add_sos=True, add_eos=True)

        if len(tokens) > self.max_len + 1:
            tokens = tokens[:self.max_len + 1]

        pad_count = (self.max_len + 1) - len(tokens)
        tokens = tokens + [self.tok.PAD_ID] * pad_count

        input_ids = tokens[:-1]
        label_ids = tokens[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label_ids, dtype=torch.long),
        }


class TrainAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class TrainDecoder(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = TrainAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.residual_connection = nn.ModuleList([ResidualConnections(d_model) for _ in range(2)])

    def forward(self, x):
        sub_layer = self.attention(x)
        x = self.residual_connection[0](x, sub_layer)
        sub_layer = self.feed_forward(x)
        x = self.residual_connection[1](x, sub_layer)
        return x


class TrainTransformer(nn.Module):

    def __init__(self, embed, pe, decoder_blocks, projection_layer):
        super().__init__()
        self.embed = embed
        self.pe = pe
        self.decoder_blocks = decoder_blocks
        self.projection_layer = projection_layer

    def forward(self, x):
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        x = self.pe(self.embed(x), position_ids)

        for block in self.decoder_blocks:
            x = block(x)

        return self.projection_layer(x)


def build_train_transformer(config, vocab_size):
    d_model = config['d_model']
    num_heads = config['num_heads']
    N = config['num_blocks']
    max_seq_len = config['tgt_max_seq_len']

    embed = Embedding(d_model, vocab_size)
    pe = PositionalEncoding(d_model, max_seq_len)

    decoder_blocks = nn.ModuleList([TrainDecoder(d_model, num_heads) for _ in range(N)])

    projection_layer = ProjectionLayer(d_model, vocab_size)

    return TrainTransformer(embed, pe, decoder_blocks, projection_layer)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)

            output = model(input_ids)
            output = output.view(-1, output.shape[-1])
            label = label.view(-1)
            loss = criterion(output, label)
            total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, criterion, optimizer, device, config):
    best_val = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)

            output = model(input_ids)
            output = output.view(-1, output.shape[-1])
            label = label.view(-1)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch+1:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'decoder_only.pt')
            print(f"  -> saved (best val so far)")


def generate(model, prompt, tok, device, max_len):
    model.eval()
    model.to(device)
    with torch.no_grad():
        tokens = tok.encode_sentence(prompt, add_sos=True, add_eos=False)
        generated_ids = list(tokens)

        for _ in range(max_len):
            input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_ids.append(next_token_id)

            if next_token_id == tok.EOS_ID:
                break

        return tok.decode_sentence(generated_ids[1:])


if __name__ == "__main__":
    sentences = load_data()
    tok = Tokenizer()
    tok.build_vocab(sentences)

    max_len = transformer_configurations['max_len']

    full_dataset = TextDataset(sentences, tok, max_len)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=transformer_configurations['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=transformer_configurations['batch_size'], shuffle=False)

    model = build_train_transformer(transformer_configurations, tok.vocab_size())

    criterion = nn.CrossEntropyLoss(ignore_index=tok.PAD_ID)
    optimizer = optim.Adam(model.parameters(), lr=transformer_configurations['learning_rate'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train(model, train_loader, val_loader, criterion, optimizer, device, transformer_configurations)
    print(generate(model, "Once upon a time", tok, device, max_len=50))