import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.multimodal_model import TransDTAP

def train_model(model, train_loader, val_loader, device, epochs=100):

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            smiles, seq, mol, prot, target = [b.to(device) for b in batch]

            optimizer.zero_grad()
            output = model(smiles, seq, mol, prot)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                smiles, seq, mol, prot, target = [b.to(device) for b in batch]
                output = model(smiles, seq, mol, prot)
                val_loss += criterion(output, target).item()

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
