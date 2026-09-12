"""
fine_tune.py

Fine-tuning مدل از پیش‌آموزش‌دیده (روی ECG-ID) با داده‌ی واقعی
جمع‌آوری‌شده از سنسور MAX30003.

روش‌شناسی ارزیابی (بدون نشت داده):
  - Train (60%)      -> آموزش مدل
  - Validation (20%) -> انتخاب بهترین Epoch و پیدا کردن Threshold بهینه
  - Test (20%)       -> فقط یک‌بار، در پایان، برای گزارش دقت نهایی
                        (با همان Threshold به‌دست‌آمده از Validation،
                        بدون هیچ جستجوی مجددی روی Test)
"""

from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

import torch
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms


# =========================================================
# ثابت کردن Seed برای تکرارپذیری کامل
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================================================
# مسیرها و تنظیمات
# =========================================================

SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))

from model import get_model

PROJECT_DIR = SRC_DIR.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED_PATH = PROJECT_DIR / "results" / "models" / "best_siamese_ecg_model_hardmining.pth"
FINETUNED_PATH = PROJECT_DIR / "results" / "models" / "finetuned_model.pth"
METADATA_FILE = PROJECT_DIR / "results" / "new_data_image_metadata.csv"
THRESHOLD_FILE = PROJECT_DIR / "results" / "unlock_threshold.txt"
FINAL_RESULT_FILE = PROJECT_DIR / "results" / "finetune_final_test_result.txt"


# =========================================================
# Hyperparameters
# =========================================================

EPOCHS = 30
LR = 1e-5
MARGIN = 2.0

# نسبت‌های تقسیم داده - جمعشان باید ۱ باشد
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

P = 4
K = 4
BATCH_SIZE = P * K

print(f"Using device: {DEVICE}")
print(f"PK Sampling: P={P}, K={K}, Batch Size={BATCH_SIZE}")


# =========================================================
# Transform
# =========================================================

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])


def load_image(path):
    img = Image.open(PROJECT_DIR / path).convert("L")
    return transform(img)


# =========================================================
# 1. خواندن Metadata
# =========================================================

df = pd.read_csv(METADATA_FILE)
persons = sorted(df["person_id"].unique())
print(f"New persons found: {persons}")

person_to_label = {person_id: i for i, person_id in enumerate(persons)}


# =========================================================
# 2. ساخت Train / Validation / Test (سه‌بخشی، بدون نشت داده)
# =========================================================

train_by_person = {}
val_records = []
test_records = []

for person_id in persons:
    person_df = df[df["person_id"] == person_id].reset_index(drop=True)

    records = [
        (row["image_path"], person_to_label[person_id])
        for _, row in person_df.iterrows()
    ]

    random.shuffle(records)

    n = len(records)
    n_test = max(1, int(n * TEST_RATIO))
    n_val = max(1, int(n * VAL_RATIO))

    # ترتیب مهم است: اول Test را جدا می‌کنیم و تا انتها دست نمی‌زنیم
    person_test = records[:n_test]
    person_val = records[n_test:n_test + n_val]
    person_train = records[n_test + n_val:]

    test_records.extend(person_test)
    val_records.extend(person_val)
    train_by_person[person_id] = person_train

train_records = []
for person_id in persons:
    train_records.extend(train_by_person[person_id])

print(
    f"Train images: {len(train_records)} | "
    f"Val images: {len(val_records)} | "
    f"Test images: {len(test_records)}  (دست‌نخورده تا پایان)"
)


# =========================================================
# بررسی حداقل داده برای PK Sampling
# =========================================================

usable_persons = [
    person_id for person_id in persons
    if len(train_by_person[person_id]) >= K
]

if len(usable_persons) < 2:
    raise RuntimeError(
        f"برای Batch-Hard Mining حداقل دو فرد با حداقل {K} تصویر Train لازم است."
    )

print(f"Usable persons for PK sampling: {usable_persons}")


# =========================================================
# 3. بارگذاری مدل
# =========================================================

model = get_model().to(DEVICE)

if not PRETRAINED_PATH.exists():
    raise FileNotFoundError(f"مدل از پیش‌آموزش‌دیده پیدا نشد:\n{PRETRAINED_PATH}")

model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
print(f"Loaded pretrained weights: {PRETRAINED_PATH}")


# =========================================================
# 4. Freeze کردن لایه‌های اولیه
# =========================================================

for layer_name in ["conv1", "bn1", "conv2", "bn2"]:
    layer = getattr(model.embedding_net, layer_name)
    for param in layer.parameters():
        param.requires_grad = False

trainable_params = [p for p in model.parameters() if p.requires_grad]
total_trainable = sum(p.numel() for p in trainable_params)
print(f"Trainable parameters (conv3 + fc only): {total_trainable:,}")

optimizer = optim.Adam(trainable_params, lr=LR)


# =========================================================
# 5. استخراج Embedding
# =========================================================

def get_embeddings(records):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for path, label in records:
            img = load_image(path).unsqueeze(0).to(DEVICE)
            emb = model.embedding_net(img)
            embeddings.append(emb.squeeze(0).cpu())
            labels.append(label)

    return torch.stack(embeddings), torch.tensor(labels)


def compute_pair_distances(records):
    """محاسبه فاصله‌ی همه‌جفت‌های ممکن در یک مجموعه (برای Val یا Test)"""
    embeddings, labels = get_embeddings(records)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    n = len(labels)
    same_dists = []
    diff_dists = []

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j].item()
            if labels[i] == labels[j]:
                same_dists.append(d)
            else:
                diff_dists.append(d)

    return np.array(same_dists), np.array(diff_dists)


# =========================================================
# 6. Evaluation روی Validation (جستجوی Threshold مجاز است)
# =========================================================

def evaluate_and_find_threshold(records, tag=""):
    if len(records) < 2:
        print(f"[{tag}] Not enough samples.")
        return 0.0, 0.5

    same_dists, diff_dists = compute_pair_distances(records)

    if len(same_dists) == 0 or len(diff_dists) == 0:
        print(f"[{tag}] جفت مثبت یا منفی کافی برای ارزیابی وجود ندارد.")
        return 0.0, 0.5

    pos_mean = same_dists.mean()
    neg_mean = diff_dists.mean()

    all_dists = np.concatenate([same_dists, diff_dists])
    all_labels = np.concatenate([np.ones(len(same_dists)), np.zeros(len(diff_dists))])

    thresholds = np.linspace(all_dists.min(), all_dists.max(), 200)
    best_acc = 0.0
    best_threshold = 0.5

    for th in thresholds:
        preds = (all_dists < th).astype(int)
        acc = np.mean(preds == all_labels) * 100
        if acc > best_acc:
            best_acc = acc
            best_threshold = th

    print(
        f"[{tag}] Positive dist avg: {pos_mean:.3f} | Negative dist avg: {neg_mean:.3f} | "
        f"Gap: {neg_mean - pos_mean:.3f} | Best Accuracy: {best_acc:.2f}% | "
        f"Best Threshold: {best_threshold:.4f}"
    )

    return best_acc, best_threshold


# =========================================================
# 7. Evaluation روی Test (Threshold از قبل مشخص است - بدون جستجو!)
# =========================================================

def evaluate_on_test(records, fixed_threshold, tag="Test"):
    if len(records) < 2:
        print(f"[{tag}] Not enough samples.")
        return 0.0, 0.0

    same_dists, diff_dists = compute_pair_distances(records)

    if len(same_dists) == 0 or len(diff_dists) == 0:
        print(f"[{tag}] جفت مثبت یا منفی کافی وجود ندارد.")
        return 0.0, 0.0

    all_dists = np.concatenate([same_dists, diff_dists])
    all_labels = np.concatenate([np.ones(len(same_dists)), np.zeros(len(diff_dists))])

    # فقط Apply می‌کنیم، جستجوی جدیدی برای Threshold انجام نمی‌دهیم
    preds = (all_dists < fixed_threshold).astype(int)
    accuracy = np.mean(preds == all_labels) * 100

    # محاسبه EER هم برای گزارش کامل‌تر
    far_list, frr_list = [], []
    for th in np.linspace(all_dists.min(), all_dists.max(), 500):
        p = (all_dists < th).astype(int)
        far = np.mean(p[all_labels == 0])
        frr = np.mean(1 - p[all_labels == 1])
        far_list.append(far)
        frr_list.append(frr)

    far_arr, frr_arr = np.array(far_list), np.array(frr_list)
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2 * 100

    print(
        f"[{tag}] با Threshold ثابت={fixed_threshold:.4f} -> "
        f"Accuracy: {accuracy:.2f}% | EER (تخمینی): {eer:.2f}%"
    )

    return accuracy, eer


# =========================================================
# 8. Batch-Hard Mining
# =========================================================

def batch_hard_mining(embeddings, labels):
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
    neg_dist[~diff_mask] = float("inf")
    hardest_neg, _ = neg_dist.min(dim=1)

    valid = (hardest_pos >= 0) & torch.isfinite(hardest_neg)

    return hardest_pos[valid], hardest_neg[valid]


def batch_hard_loss(hardest_pos, hardest_neg, margin=MARGIN):
    pos_loss = hardest_pos.pow(2)
    neg_loss = torch.clamp(margin - hardest_neg, min=0.0).pow(2)
    return (pos_loss + neg_loss).mean()


# =========================================================
# 9. ساخت PK Batch (فقط از Train)
# =========================================================

def create_pk_batch():
    selected_persons = random.sample(usable_persons, min(P, len(usable_persons)))

    batch = []
    for person_id in selected_persons:
        person_records = train_by_person[person_id]
        selected_records = random.sample(person_records, K)
        batch.extend(selected_records)

    random.shuffle(batch)
    return batch


# =========================================================
# 10. ارزیابی Zero-shot قبل از Fine-tuning (روی Val، فقط برای مقایسه)
# =========================================================

print("\n--- ارزیابی قبل از Fine-tuning (Zero-shot روی Validation) ---")
evaluate_and_find_threshold(val_records, tag="Before")


# =========================================================
# 11. حلقه‌ی Fine-tuning
# =========================================================

print("\n--- شروع Fine-tuning ---")

best_val_acc = 0.0
best_threshold = 0.5

STEPS_PER_EPOCH = max(20, len(train_records) // BATCH_SIZE)
print(f"Steps per epoch: {STEPS_PER_EPOCH}")

for epoch in range(EPOCHS):
    model.train()
    model.embedding_net.bn1.eval()
    model.embedding_net.bn2.eval()

    epoch_losses = []

    for step in range(STEPS_PER_EPOCH):
        batch = create_pk_batch()

        imgs = torch.stack([load_image(path) for path, _ in batch]).to(DEVICE)
        labels = torch.tensor([label for _, label in batch]).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        embeddings = model.embedding_net(imgs)

        hardest_pos, hardest_neg = batch_hard_mining(embeddings, labels)

        if len(hardest_pos) == 0:
            continue

        loss = batch_hard_loss(hardest_pos, hardest_neg)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        del imgs, labels, embeddings, hardest_pos, hardest_neg, loss

    mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}] Loss: {mean_loss:.4f}")

    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        # مهم: انتخاب مدل و Threshold فقط بر اساس Validation - نه Test!
        val_acc, val_threshold = evaluate_and_find_threshold(val_records, tag=f"Epoch {epoch + 1}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_threshold = val_threshold

            FINETUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), FINETUNED_PATH)
            print(f"--> Saved best fine-tuned model (Val Accuracy: {best_val_acc:.2f}%)")


# =========================================================
# 12. ارزیابی نهایی روی Test (فقط یک‌بار، در انتها!)
# =========================================================

print("\n" + "=" * 60)
print("ارزیابی نهایی روی Test Set (دست‌نخورده تا این لحظه)")
print("=" * 60)

# بارگذاری بهترین مدل ذخیره‌شده
model.load_state_dict(torch.load(FINETUNED_PATH, map_location=DEVICE))

test_accuracy, test_eer = evaluate_on_test(
    test_records,
    fixed_threshold=best_threshold,
    tag="Final Test"
)


# =========================================================
# 13. ذخیره‌ی نتایج نهایی
# =========================================================

with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
    f.write(f"threshold={best_threshold:.4f}\n")
    f.write(f"val_accuracy={best_val_acc:.2f}\n")
    f.write(f"test_accuracy={test_accuracy:.2f}\n")
    f.write(f"test_eer={test_eer:.2f}\n")

with open(FINAL_RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("=== نتیجه نهایی Fine-tuning روی داده MAX30003 ===\n")
    f.write(f"تعداد افراد: {len(persons)}\n")
    f.write(f"Train images: {len(train_records)}\n")
    f.write(f"Val images: {len(val_records)}\n")
    f.write(f"Test images: {len(test_records)}\n")
    f.write(f"Threshold بهینه (از روی Validation): {best_threshold:.4f}\n")
    f.write(f"دقت نهایی روی Test: {test_accuracy:.2f}%\n")
    f.write(f"EER نهایی روی Test: {test_eer:.2f}%\n")

print(f"\nBest fine-tuned model: {FINETUNED_PATH}")
print(f"Threshold + نتایج ذخیره شد در: {THRESHOLD_FILE}")
print(f"گزارش کامل ذخیره شد در: {FINAL_RESULT_FILE}")
print(f"\n>>> عدد نهایی برای گزارش پروژه: Test Accuracy = {test_accuracy:.2f}% | Test EER = {test_eer:.2f}% <<<")