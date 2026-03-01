import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            smiles, seq, mol, prot, target = [b.to(device) for b in batch]
            output = model(smiles, seq, mol, prot)
            preds.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    r2 = r2_score(targets, preds)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)

    print(f"R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
