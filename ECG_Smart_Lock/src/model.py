import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGEmbeddingCNN(nn.Module):
    def __init__(self):
        super(ECGEmbeddingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc = nn.Linear(128 * 14 * 14, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 14 * 14)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNetwork(nn.Module):
    """معماری شبکه سیامز برای مقایسه جفت تصاویر نوار قلب"""
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = ECGEmbeddingCNN()

    def forward(self, input1, input2):
        # استخراج بردار ویژگی برای هر دو تصویر ورودی به صورت کاملاً مستقل و ایمن
        out1 = self.embedding_net(input1)
        out2 = self.embedding_net(input2)
        return out1, out2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1.0 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


def get_model(num_classes=None):
    return SiameseNetwork()