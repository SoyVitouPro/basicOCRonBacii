import os
import json
import heapq
import torch
import torch.optim as optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

# Local imports
from dataloaders import OCRVocab, OCRJsonDataset, ctc_collate
from modelCRNN import CRNN


# -----------------------------
# Config
# -----------------------------
dataset_path = "/data/Vitou/gen_data/meta_data.json"
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
batch_size = 32
learning_rate = 1e-3

# -----------------------------
# Build Vocab
# -----------------------------
with open(dataset_path, encoding="utf-8") as f:
    data = json.load(f)
all_labels = [d["label"] for d in data]

vocab = OCRVocab()
vocab.build_from_labels(all_labels)

# -----------------------------
# Dataset & Loader
# -----------------------------
dataset = OCRJsonDataset(dataset_path, vocab=vocab)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=4, collate_fn=ctc_collate)

# Quick sanity check
batch = next(iter(loader))
print("Images:", batch["images"].shape)        # [B, 1, 32, 128]
print("Labels:", batch["labels"].shape)        # [B, Lmax]
print("Lengths:", batch["label_lengths"])      # [B]

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = CRNN(img_h=32, num_classes=vocab.num_classes).to(device)
criterion = CTCLoss(blank=OCRVocab.PAD, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Track best models
best_models = []  # min-heap: stores (-loss, path)


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"]

        # Forward
        logits = model(images)  # [T, B, C]
        input_lengths = torch.full(
            size=(logits.size(1),),  # B
            fill_value=logits.size(0),
            dtype=torch.long,
        )

        # Loss
        loss = criterion(logits, labels, input_lengths, label_lengths)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # -----------------------------
    # Save Last Model
    # -----------------------------
    last_model_path = os.path.join(save_dir, "model_last.pth")
    torch.save({
        "epoch": epoch+1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": avg_loss,
        "vocab": vocab.chars,
    }, last_model_path)

    # -----------------------------
    # Save Best 2 Models
    # -----------------------------
    checkpoint_path = os.path.join(save_dir, f"tmp_epoch{epoch+1}.pth")
    torch.save({
        "epoch": epoch+1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": avg_loss,
        "vocab": vocab.chars,
    }, checkpoint_path)

    # Push into heap
    heapq.heappush(best_models, (-avg_loss, checkpoint_path))

    if len(best_models) > 2:
        _, worst_path = heapq.heappop(best_models)
        if os.path.exists(worst_path):
            os.remove(worst_path)


# --- At the end, rename the 2 best models clearly ---
best_models = sorted(best_models, reverse=True)  # lowest loss first
for i, (_, path) in enumerate(best_models, 1):
    new_path = os.path.join(save_dir, f"model_best{i}.pth")
    os.rename(path, new_path)
    print(f"Saved best model {i}: {new_path}")