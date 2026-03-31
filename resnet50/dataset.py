import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

# data_dir=r'C:\Users\Lenovo\Desktop\0109test'

def get_loaders(data_dir=r'/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/generated_enhanced_6145', batch_size=64, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    将数据集划分为训练集、验证集、测试集
    train_ratio + val_ratio + test_ratio = 1.0

    参数:
        data_dir: 数据目录路径
        batch_size: 批大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 验证比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "训练、验证、测试集比例之和必须为1"

    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((64,64)),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomResizedCrop(64),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 验证集和测试集的transform（相同）
    val_test_transform = transforms.Compose([
        transforms.Resize((64,64)),
        # transforms.CenterCrop(224),
        # transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 加载完整数据集
    full_dataset = datasets.ImageFolder(data_dir)

    # 打印数据集信息
    print("=" * 50)
    print("📂 Dataset Information:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Number of classes: {len(full_dataset.classes)}")
    print(f"  Class names: {full_dataset.classes}")
    print("=" * 50)

    # 计算各集合的大小
    n_total = len(full_dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val  # 确保总数正确

    # 固定随机种子以保证可重复性
    generator = torch.Generator().manual_seed(40)

    # 随机划分数据集
    train_ds, temp_ds = random_split(
        full_dataset,
        [n_train, n_val + n_test],
        generator=generator
    )

    # 从剩余数据中划分验证集和测试集
    val_ds, test_ds = random_split(
        temp_ds,
        [n_val, n_test],
        generator=generator
    )

    # 方法1：修改子集的transform（简单但有效）
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_test_transform
    test_ds.dataset.transform = val_test_transform

    # 方法2：使用自定义Dataset类（更安全）
    # class TransformedSubset(torch.utils.data.Dataset):
    #     """应用transform的子集"""
    #
    #     def __init__(self, subset, transform=None):
    #         self.subset = subset
    #         self.transform = transform
    #
    #     def __getitem__(self, index):
    #         x, y = self.subset[index]
    #         if self.transform:
    #             x = self.transform(x)
    #         return x, y
    #
    #     def __len__(self):
    #         return len(self.subset)

    # 如果需要使用方法2，取消注释下面的代码
    # train_ds = TransformedSubset(train_ds, transform=train_transform)
    # val_ds = TransformedSubset(val_ds, transform=val_test_transform)
    # test_ds = TransformedSubset(test_ds, transform=val_test_transform)

    # 创建DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    class_names = os.listdir(data_dir)

    # 打印划分统计信息
    print(f"\n📊 Dataset Split Statistics:")
    print(f"  Training set:   {len(train_ds)} samples ({len(train_ds) / n_total * 100:.1f}%)")
    print(f"  Validation set: {len(val_ds)} samples ({len(val_ds) / n_total * 100:.1f}%)")
    print(f"  Test set:       {len(test_ds)} samples ({len(test_ds) / n_total * 100:.1f}%)")
    print("=" * 50)

    return train_loader, val_loader, test_loader, class_names


def get_classes(data_dir=r'/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/generated_enhanced_6145'):
    return os.listdir(data_dir), len(os.listdir(data_dir))

