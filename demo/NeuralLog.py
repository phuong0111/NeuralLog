import os
import sys

sys.path.append("../")

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.utils import shuffle
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import load_supercomputers
from utils import report, EarlyStopping

log_file = "logs/BGL.log"
embed_dim = 768  # Embedding size for each token
max_len = 75
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
            x = x[-self.max_len:]
        
        # Pad if too short
        if len(x) < self.max_len:
            padding = np.zeros((self.max_len - len(x), self.embed_dim))
            x = np.concatenate([padding, x])
            
        x = torch.FloatTensor(x)
        y = torch.LongTensor([self.Y[idx]])
        return x, y.squeeze()

def train_model(model, train_loader, val_loader, num_epochs, device, model_path, patience=5):
    """Training loop with validation"""
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # Warm-up period
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
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
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{total_loss/len(train_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_x, batch_y in val_pbar:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = F.cross_entropy(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss/len(val_loader):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100.*val_correct/val_total:.2f}%')
        
        # Early stopping
        val_loss = val_loss/len(val_loader)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        # Save best model
        val_acc = 100. * val_correct/val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_path)
    
    # Load best model
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

def test_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc='Testing'):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    # Print classification report
    print(report(all_labels, all_preds))
    return all_preds, all_labels

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    log_file = "../logs/BGL.log"
    train_loader, test_loader, (x_train, y_train), (x_test, y_test) = load_supercomputers(
        log_file,
        train_ratio=0.8,
        windows_size=20,
        step_size=20,
        e_type='bert',
        mode='balance',
        batch_size=256
    )
    
    # Initialize model
    from models import TransformerClassifier  # Import your PyTorch model
    model = TransformerClassifier(
        embed_dim=768,
        ff_dim=2048,
        max_len=75,
        num_heads=12,
        dropout=0.1
    ).to(device)
    
    # Train model
    print("Training model...")
    model_path = "bgl_transformer.pt"
    model = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=10,
        device=device,
        model_path=model_path
    )
    
    # Test model
    print("Testing model...")
    test_model(model, test_loader, device)

if __name__ == '__main__':
    main()