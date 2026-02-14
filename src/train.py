"""
Training utilities.
"""

import time
import torch
import torch.nn as nn


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for images, labels, lengths in loader:
        images = images.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)  # (seq_len, batch, vocab)
        
        # CTC loss needs input_lengths
        input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)
        
        loss = criterion(outputs, labels.cpu(), input_lengths, lengths.cpu())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, labels, lengths in loader:
            images = images.to(device)
            outputs = model(images)
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)
            loss = criterion(outputs, labels, input_lengths, lengths)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def fit(model, train_loader, val_loader, epochs=80, lr=1e-3, device="cuda"):
    """
    Full training loop.
    
    Returns:
        train_losses, val_losses
    """
    criterion = nn.CTCLoss(blank=1, zero_infinity=True)  # blank = '-' index
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.1)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {time.time()-start:.1f}s")
    
    return train_losses, val_losses
