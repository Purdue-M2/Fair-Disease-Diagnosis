"""Training script for Fair Disease Diagnosis (standalone / vast.ai)."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from src.dataset import LungCTDataset
from src.model import LungDiagnosisModel3D
from src.losses import FairLACVaRLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----------------------------------------------------------
    train_dirs = [
        (os.path.join(args.data_dir, "Train/adenocarcinoma"), 0),
        (os.path.join(args.data_dir, "Train/squamous_cell"), 1),
        (os.path.join(args.data_dir, "Train/covid19"), 2),
        (os.path.join(args.data_dir, "Train/normal"), 3),
    ]
    val_dirs = [
        (os.path.join(args.data_dir, "Val/adenocarcinoma"), 0),
        (os.path.join(args.data_dir, "Val/squamous_cell"), 1),
        (os.path.join(args.data_dir, "Val/covid19"), 2),
        (os.path.join(args.data_dir, "Val/normal"), 3),
    ]

    train_dataset = LungCTDataset(train_dirs, is_train=True)
    val_dataset = LungCTDataset(val_dirs, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---- Model ---------------------------------------------------------
    model = LungDiagnosisModel3D(num_classes=4).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Loss / Optimiser ----------------------------------------------
    class_counts = [0, 0, 0, 0]
    for _, label, _ in train_dataset.samples:
        class_counts[label] += 1

    criterion = FairLACVaRLoss(class_counts, alpha=args.alpha, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # ---- Training loop -------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    best_f1, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for volumes, labels, genders in train_loader:
            volumes, labels, genders = volumes.to(device), labels.to(device), genders.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(volumes)
                loss = criterion(outputs, labels, genders)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        scheduler.step()
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)

        # ---- Validation ------------------------------------------------
        model.eval()
        val_preds, val_labels, val_genders_list = [], [], []
        with torch.no_grad():
            for volumes, labels, genders in val_loader:
                volumes = volumes.to(device)
                with autocast():
                    outputs = model(volumes)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.numpy())
                val_genders_list.extend(genders.numpy())

        macro_f1 = f1_score(val_labels, val_preds, average="macro")
        male_mask = [g == 1 for g in val_genders_list]
        female_mask = [g == 0 for g in val_genders_list]
        f1_m = f1_score(
            [l for l, m in zip(val_labels, male_mask) if m],
            [p for p, m in zip(val_preds, male_mask) if m],
            average="macro",
        )
        f1_f = f1_score(
            [l for l, m in zip(val_labels, female_mask) if m],
            [p for p, m in zip(val_preds, female_mask) if m],
            average="macro",
        )
        fair_score = 0.5 * (f1_m + f1_f)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Loss: {avg_loss:.4f} | Acc: {train_acc:.1f}% | "
            f"F1: {macro_f1:.4f} | M: {f1_m:.4f} | F: {f1_f:.4f} | "
            f"Gap: {abs(f1_m - f1_f):.4f} | Score: {fair_score:.4f}"
        )

        if fair_score > best_f1:
            best_f1 = fair_score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model_fair.pth"))

    print(f"\nBest epoch: {best_epoch} | Best fair score: {best_f1:.4f}")
    print(f"Model saved to {args.save_dir}/best_model_fair.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair Disease Diagnosis Training")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/extracted")
    parser.add_argument("--save_dir", type=str, default="/workspace/outputs")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
