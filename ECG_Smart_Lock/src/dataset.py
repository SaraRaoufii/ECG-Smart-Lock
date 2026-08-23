import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import itertools

# مسیر اصلی پروژه برای بارگذاری ایمن تصاویر
PROJECT_DIR = Path(__file__).resolve().parent.parent

class SiameseECGDataset(Dataset):
    def __init__(self, metadata_df, split='train', transform=None):
        self.transform = transform
        self.split = split
        
        # فیلتر کردن دیتافریم بر اساس نوع بخش (train, validation, test)
        self.df = metadata_df[metadata_df['split'] == split].reset_index(drop=True)
        
        # دسته‌بندی مسیر تصاویر بر اساس person_id
        self.person_to_images = {}
        for _, row in self.df.iterrows():
            p_id = row['person_id']
            img_path = row['image_path']
            if p_id not in self.person_to_images:
                self.person_to_images[p_id] = []
            self.person_to_images[p_id].append(img_path)
            
        self.persons = list(self.person_to_images.keys())
        
        # ساخت لیست ثابتِ جفت‌ها در حافظه (Pairs List با کنترل توازن)
        self.pairs = self._create_balanced_pairs()

    def resample_pairs(self):
        """جفت‌ها را دوباره تصادفی می‌سازد - در ابتدای هر epoch صدا زده شود"""
        self.pairs = self._create_balanced_pairs()

    def _create_balanced_pairs(self):
        pairs = []
        
        for person_id, img_paths in self.person_to_images.items():
            if len(img_paths) < 2:
                continue
                
            # جلوگیری از رشد درجه دوم و ناموازنی: نمونه‌گیری محدود از جفت‌های مثبت
            all_pos_pairs = list(itertools.combinations(img_paths, 2))
            max_pos_per_person = min(len(all_pos_pairs), len(img_paths) * 3)  # حداکثر ۳ برابر تعداد تصاویر
            sampled_pos = random.sample(all_pos_pairs, max_pos_per_person)
            
            for p1, p2 in sampled_pos:
                pairs.append((p1, p2, 1.0)) # لیبل 1: شخص یکسان
                
            # ساخت جفت‌های منفی (Imposter) متناسب با تعداد مثبت‌ها
            other_persons = [p for p in self.persons if p != person_id]
            if not other_persons:
                continue
                
            for _ in range(len(sampled_pos) * 2):
                img_path = random.choice(img_paths)
                neg_person = random.choice(other_persons)
                neg_img_path = random.choice(self.person_to_images[neg_person])
                pairs.append((img_path, neg_img_path, 0.0)) # لیبل 0: شخص متفاوت
                
        # مخلوط کردن کل جفت‌ها برای جلوگیری از الگوی ترتیبی
        random.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx): # <-- اصلاح مشکل ۱: اضافه شدن خودکار self
        img1_path, img2_path, label = self.pairs[idx]
        
        # ترکیب ایمن مسیر با PROJECT_DIR برای جلوگیری از خطای FileNotFoundError
        full_img1_path = PROJECT_DIR / img1_path
        full_img2_path = PROJECT_DIR / img2_path
        
        img1 = Image.open(full_img1_path).convert('L')
        img2 = Image.open(full_img2_path).convert('L')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(batch_size=32):
    # اصلاح مسیر برای خواندن صحیح فایل از پوشه results
    csv_path = PROJECT_DIR / "results" / "image_metadata.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"فایل در این مسیر یافت نشد: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    
    train_dataset = SiameseECGDataset(df, split='train', transform=transform)
    val_dataset = SiameseECGDataset(df, split='validation', transform=transform)
    test_dataset = SiameseECGDataset(df, split='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_persons = len(df['person_id'].unique())
    
    return train_loader, val_loader, test_loader, num_persons


from torch.utils.data import Sampler

class SingleImageECGDataset(Dataset):
    """دیتاست تک‌تصویری - برای PK-Sampling در آموزش Batch-Hard"""
    def __init__(self, metadata_df, split='train', transform=None):
        self.transform = transform
        self.df = metadata_df[metadata_df['split'] == split].reset_index(drop=True)

        self.persons = sorted(self.df['person_id'].unique())
        self.person_to_label = {p: i for i, p in enumerate(self.persons)}

        self.person_to_indices = {}
        for idx, row in self.df.iterrows():
            self.person_to_indices.setdefault(row['person_id'], []).append(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(PROJECT_DIR / row['image_path']).convert('L')
        if self.transform:
            img = self.transform(img)
        label = self.person_to_label[row['person_id']]
        return img, label


class PKBatchSampler(Sampler):
    """هر Batch را از P نفر، K تصویر به‌ازای هر نفر می‌سازد"""
    def __init__(self, person_to_indices, P=8, K=4):
        self.person_to_indices = {p: idxs for p, idxs in person_to_indices.items() if len(idxs) >= 2}
        self.P = min(P, len(self.person_to_indices))
        self.K = K
        self.persons = list(self.person_to_indices.keys())
        total_images = sum(len(v) for v in self.person_to_indices.values())
        self.num_batches = max(1, total_images // (self.P * self.K))

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_persons = random.sample(self.persons, self.P)
            batch_indices = []
            for p in batch_persons:
                indices = self.person_to_indices[p]
                if len(indices) >= self.K:
                    chosen = random.sample(indices, self.K)
                else:
                    chosen = random.choices(indices, k=self.K)  # نمونه‌ی تکراری اگه کم داشت
                batch_indices.extend(chosen)
            yield batch_indices

    def __len__(self):
        return self.num_batches


def get_train_pk_loader(P=8, K=4, transform=None):
    csv_path = PROJECT_DIR / "results" / "image_metadata.csv"
    df = pd.read_csv(csv_path)
    dataset = SingleImageECGDataset(df, split='train', transform=transform)
    sampler = PKBatchSampler(dataset.person_to_indices, P=P, K=K)
    return DataLoader(dataset, batch_sampler=sampler)