import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, accuracy_score
)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Validate Discriminator on a dataset")
    parser.add_argument('-m', '--model', required=True, type=str, help="Path to discriminator .pth file")
    parser.add_argument('-d', '--data', required=True, type=str, help="Path to validation dataset (ImageFolder)")
    parser.add_argument('-o', '--output', required=True, type=str, help="Folder to save validation results")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def validate_discriminator(model_path, data_path, output_path, batch_size, device):
    os.makedirs(output_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    class_names = dataset.classes  # Assumes: 0=real, 1=fake or viceversa
    print(f"Class mapping: {class_names}")

    # Load model
    netD = torch.load(model_path, map_location=device)
    netD.eval()

    all_labels = []
    all_scores = []
    all_preds = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            outputs = netD(inputs).view(-1)
            loss = criterion(outputs, labels)

            preds = (outputs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / total_samples

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]

    print(f"Validation Results:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}, EER: {eer:.4f}")

    # Save metrics table
    metrics = {
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1],
        "AUC": [auc],
        "Accuracy": [acc],
        "Loss": [avg_loss],
        "EER": [eer]
    }

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_path, "validation_metrics.csv"), index=False)
    df.to_string(buf=open(os.path.join(output_path, "validation_metrics.txt"), "w"), index=False)

    # Save ROC Curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "roc_curve.png"))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    validate_discriminator(args.model, args.data, args.output, args.batch_size, args.device)
