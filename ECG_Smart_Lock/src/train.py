from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from dataset import get_dataloaders, get_train_pk_loader
from model import get_model

BATCH_P = 8       # تعداد افراد در هر Batch
BATCH_K = 4       # تعداد تصویر از هر فرد  → Batch size = 32
LEARNING_RATE = 0.001
EPOCHS = 60
MARGIN = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def get_distances_labels(model, data_loader):
    model.eval()
    distances, labels_list = [], []
    with torch.no_grad():
        for img1, img2, labels in data_loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            output1, output2 = model(img1, img2)
            dist = nn.functional.pairwise_distance(output1, output2)
            distances.extend(dist.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    return np.array(distances), np.array(labels_list)


def find_best_threshold(distances, labels):
    thresholds = np.linspace(distances.min(), distances.max(), num=100)
    best_acc, best_th = 0.0, 0.5
    for th in thresholds:
        preds = (distances < th).astype(int)
        acc = np.mean(preds == labels) * 100
        if acc > best_acc:
            best_acc, best_th = acc, th
    return best_acc, best_th


def compute_eer(distances, labels):
    """محاسبه نرخ خطای برابر (EER)؛ labels: 1=genuine(similar), 0=impostor(different)"""
    thresholds = np.linspace(distances.min(), distances.max(), 1000)
    fars, frrs = [], []
    for th in thresholds:
        preds_genuine = distances < th
        far = np.mean(preds_genuine[labels == 0])   # پذیرش اشتباه غریبه
        frr = np.mean(~preds_genuine[labels == 1])  # رد اشتباه فرد واقعی
        fars.append(far)
        frrs.append(frr)
    fars, frrs = np.array(fars), np.array(frrs)
    idx = np.argmin(np.abs(fars - frrs))
    return (fars[idx] + frrs[idx]) / 2, thresholds[idx]


def batch_hard_mining(embeddings, labels):
    """برای هر anchor، سخت‌ترین مثبت (دورترین هم‌نفر) و سخت‌ترین منفی (نزدیک‌ترین غریبه) را پیدا می‌کند"""
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    labels_col = labels.unsqueeze(0)
    same_mask = (labels_col == labels_col.t())

    B = embeddings.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    same_no_self = same_mask & (~eye)
    diff_mask = ~same_mask

    pos_dist = dist_matrix.clone()
    pos_dist[~same_no_self] = -1.0
    hardest_pos, _ = pos_dist.max(dim=1)

    neg_dist = dist_matrix.clone()
    neg_dist[~diff_mask] = float('inf')
    hardest_neg, _ = neg_dist.min(dim=1)

    valid = hardest_pos >= 0  # anchorهایی که حداقل یک هم‌نفر دیگر در batch دارند
    return hardest_pos[valid], hardest_neg[valid]


def batch_hard_loss(hardest_pos, hardest_neg, margin=MARGIN):
    pos_loss = hardest_pos.pow(2)
    neg_loss = torch.clamp(margin - hardest_neg, min=0.0).pow(2)
    return (pos_loss + neg_loss).mean()


def diagnostic_check(model, val_loader):
    distances, labels = get_distances_labels(model, val_loader)
    pos_mean = distances[labels == 1].mean()
    neg_mean = distances[labels == 0].mean()
    gap = neg_mean - pos_mean
    print(f"   [Diagnostic] Positive dist avg: {pos_mean:.3f} | "
          f"Negative dist avg: {neg_mean:.3f} | Gap: {gap:.3f}")
    return gap


def main():
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0)),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    print("Loading datasets...")
    train_pk_loader = get_train_pk_loader(P=BATCH_P, K=BATCH_K, transform=train_transform)
    _, val_loader, test_loader, num_persons = get_dataloaders(batch_size=32)
    print(f"Total Unique Persons: {num_persons}")
    print(f"Batches per epoch (PK sampling): {len(train_pk_loader)}")

    model = get_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    best_val_acc = 0.0
    models_dir = Path(__file__).resolve().parent.parent / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_siamese_ecg_model_hardmining.pth"

    print("\n--- Starting Batch-Hard Mining Training ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_pk_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            embeddings = model.embedding_net(imgs)   # مستقیم از زیرشبکه Embedding استفاده می‌کنیم

            hardest_pos, hardest_neg = batch_hard_mining(embeddings, labels)
            if len(hardest_pos) == 0:
                continue

            loss = batch_hard_loss(hardest_pos, hardest_neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / max(1, n_batches)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Summary -> Train Loss: {epoch_loss:.4f}")

        if (epoch + 1) % 3 == 0 or epoch == EPOCHS - 1:
            diagnostic_check(model, val_loader)
            val_distances, val_labels = get_distances_labels(model, val_loader)
            val_acc, _ = find_best_threshold(val_distances, val_labels)
            print(f"   [Validation Accuracy]: {val_acc:.2f}%")
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"--> Saved Best Model with Val Accuracy: {best_val_acc:.2f}%")

    print("\nTraining Finished.")
    print(f"Best model saved at: {best_model_path}")

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(best_model_path))

    val_distances, val_labels = get_distances_labels(model, val_loader)
    _, optimal_threshold = find_best_threshold(val_distances, val_labels)
    print(f"Optimal Threshold (Validation): {optimal_threshold:.4f}")

    test_distances, test_labels = get_distances_labels(model, test_loader)
    test_preds = (test_distances < optimal_threshold).astype(int)
    test_acc = np.mean(test_preds == test_labels) * 100
    print(f"Final Test Verification Accuracy: {test_acc:.2f}%")

    # محاسبه و چاپ EER نهایی روی مجموعه تست
    eer, eer_threshold = compute_eer(test_distances, test_labels)
    print(f"Final Test EER: {eer*100:.2f}% at threshold {eer_threshold:.4f}")


if __name__ == "__main__":
    main()