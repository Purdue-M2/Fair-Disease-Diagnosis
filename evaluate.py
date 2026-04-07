"""Evaluation script with gender-stratified metrics and visualisation."""

import argparse
import os

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from dataset import LungCTDataset
from model import LungDiagnosisModel3D


CLASS_NAMES = ["Adenocarcinoma", "Squamous Cell", "COVID-19", "Normal"]


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dirs = [
        (os.path.join(args.data_dir, "Val/adenocarcinoma"), 0),
        (os.path.join(args.data_dir, "Val/squamous_cell"), 1),
        (os.path.join(args.data_dir, "Val/covid19"), 2),
        (os.path.join(args.data_dir, "Val/normal"), 3),
    ]
    val_dataset = LungCTDataset(val_dirs, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = LungDiagnosisModel3D(num_classes=4).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs, all_genders = [], [], [], []

    with torch.no_grad():
        for volumes, labels, genders in val_loader:
            volumes = volumes.to(device)
            with autocast():
                outputs = model(volumes)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_genders.extend(genders.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_genders = np.array(all_genders)

    # ---- Overall -------------------------------------------------------
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # ---- Gender-stratified ---------------------------------------------
    male = all_genders == 1
    female = all_genders == 0
    f1_m = f1_score(all_labels[male], all_preds[male], average="macro")
    f1_f = f1_score(all_labels[female], all_preds[female], average="macro")
    fair_score = 0.5 * (f1_m + f1_f)

    print("=" * 60)
    print("GENDER-WISE EVALUATION (FAIRNESS)")
    print("=" * 60)
    print(f"Male   — Macro F1: {f1_m:.4f}  (n={male.sum()})")
    print(f"Female — Macro F1: {f1_f:.4f}  (n={female.sum()})")
    print(f"F1 Gap:            {abs(f1_m - f1_f):.4f}")
    print(f"Fair Score:        {fair_score:.4f}")

    # ---- Plots ---------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    im = axes[0].imshow(cm, cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix", fontsize=14)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axes[0].set_yticklabels(CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

    # ROC curves
    labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "evaluation_plots.png"), dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to {args.output_dir}/evaluation_plots.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/workspace/data/extracted")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/workspace/outputs")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    evaluate(args)
