import os
import sys

sys.path.append("../")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.utils import shuffle
import torch.nn.functional as F
from tqdm import tqdm

from neurallog.data_loader import load_supercomputers
from neurallog.utils import report, EarlyStopping


class LogDataset(Dataset):
    def __init__(self, X, Y, max_len=75, embed_dim=768):
        self.X = X
        self.Y = Y
        self.max_len = max_len
        self.embed_dim = embed_dim

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        # Truncate if too long
        if len(x) > self.max_len:
            x = x[-self.max_len :]

        # Pad if too short
        if len(x) < self.max_len:
            padding = np.zeros((self.max_len - len(x), self.embed_dim))
            x = np.concatenate([padding, x])

        x = torch.FloatTensor(x)
        y = torch.LongTensor([self.Y[idx]])
        return x, y.squeeze()


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    train_pbar = tqdm(train_loader, desc="Training")
    for batch_x, batch_y in train_pbar:
        optimizer.zero_grad()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = F.cross_entropy(outputs, batch_y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

        train_pbar.set_postfix(
            {
                "loss": f"{total_loss/len(train_loader):.4f}",
                "acc": f"{100.*correct/total:.2f}%",
            }
        )

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validating")
        for batch_x, batch_y in val_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, batch_y)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            val_pbar.set_postfix(
                {
                    "loss": f"{total_loss/len(val_loader):.4f}",
                    "acc": f"{100.*correct/total:.2f}%",
                }
            )

    return total_loss / len(val_loader), 100.0 * correct / total


def train_model(
    model, train_loader, val_loader, num_epochs, device, model_path, patience=5
):
    """Training loop with validation"""
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)

    # Load checkpoint if exists
    start_epoch = 0
    best_val_loss = float("inf")
    checkpoint = None
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Save if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            torch.save(state, model_path)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return model


def test_model(model, test_loader, device):
    """Test the model"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    print("\nTest Results:")
    print(report(all_labels, all_preds))
    return all_preds, all_labels


def main():
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    log_file = "../logs/BGL.log"
    train_loader, test_loader, (x_train, y_train), (x_test, y_test) = (
        load_supercomputers(
            log_file,
            train_ratio=0.8,
            windows_size=20,
            step_size=20,
            e_type="bert",
            mode="balance",
            batch_size=256,
        )
    )

    # Initialize model
    from neurallog.models.transformers import TransformerClassifier

    model = TransformerClassifier(
        embed_dim=768, ff_dim=2048, max_len=75, num_heads=12, dropout=0.1
    ).to(device)

    model_path = "bgl_transformer.pth"

    # Train model
    print("Training model...")
    model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=2,
        device=device,
        model_path=model_path,
    )

    # # Test model
    # print("Testing model...")
    # test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
