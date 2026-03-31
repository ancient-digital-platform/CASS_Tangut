import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from model import EfficientNetModel
from dataset import get_loaders
import time
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import argparse


def calculate_topk_accuracy(outputs, labels, k=5):
    """计算top-k准确率"""
    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    correct_k = correct[:, :k].reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / labels.size(0))


def calculate_mAP(outputs, labels, num_classes):
    """计算mAP并统计没有正样本的类别"""
    # 将logits转换为概率
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 创建one-hot编码
    labels_one_hot = np.zeros((len(labels_np), num_classes))
    labels_one_hot[np.arange(len(labels_np)), labels_np] = 1

    # 统计
    no_positives_classes = []  # 没有正样本的类别
    valid_aps = []  # 有效的AP值

    print("📊 正样本统计:")
    for class_idx in range(num_classes):
        y_true = labels_one_hot[:, class_idx]
        n_positives = int(np.sum(y_true))

        if n_positives == 0:
            no_positives_classes.append(class_idx)
            print(f"  ❌ 类别 {class_idx}: 0个正样本")
        else:
            y_score = probs[:, class_idx]
            try:
                ap = average_precision_score(y_true, y_score)
                if not np.isnan(ap):
                    valid_aps.append(ap)
                    print(f"  ✅ 类别 {class_idx}: {n_positives}个正样本, AP={ap:.4f}")
                else:
                    print(f"  ⚠️  类别 {class_idx}: AP计算为NaN")
            except:
                print(f"  ❌ 类别 {class_idx}: AP计算失败")

    # 汇总
    print(f"\n📈 统计汇总:")
    print(f"  总类别数: {num_classes}")
    print(f"  有正样本的类别: {num_classes - len(no_positives_classes)}")
    print(f"  没有正样本的类别: {len(no_positives_classes)}")

    if no_positives_classes:
        print(f"  无正样本的类别索引: {no_positives_classes}")

    # 计算mAP
    if valid_aps:
        mAP = np.mean(valid_aps) * 100
    else:
        mAP = 0.0
        print("  ⚠️  警告: 没有任何类别的AP可计算")

    print(f"  可计算AP的类别数: {len(valid_aps)}")
    print(f"  最终mAP: {mAP:.2f}%")

    return mAP

def train(data_dir, dataset_name):
    # 训练参数配置
    log_interval = 10  # 每10个batch打印一次日志
    num_epochs = 20
    best_acc = 0
    start_time = time.time()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 加载数据
    train_loader, val_loader, test_loader, class_names = get_loaders(data_dir)
    num_classes = len(class_names)

    print(f"📊 数据集信息: 训练集={len(train_loader.dataset)} 验证集={len(val_loader.dataset)} "
          f"测试集={len(test_loader.dataset)} 类别数={num_classes}")

    # 模型初始化
    model = EfficientNetModel(num_classes=num_classes, pretrained=True).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_m = total_params / 1_000_000  # 转换为M单位

    print(f"📈 模型参数量: {params_in_m:.2f}M (可训练: {trainable_params / 1_000_000:.2f}M)")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, eps=1e-3)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    # 混合精度训练设置
    use_amp = str(device) == 'cuda'
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # 创建输出目录
    output_dir = f"outputs-{dataset_name}-0228"
    os.makedirs(output_dir, exist_ok=True)

    # 训练循环
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # ==================== 训练阶段 ====================
        model.train()
        total_loss = 0
        train_preds = []
        train_trues = []
        train_probs = []
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n{'=' * 60}")
        print(f"📅 Epoch {epoch}/{num_epochs} | LR: {current_lr:.6f} | 参数量: {params_in_m:.2f}M")
        print(f"{'=' * 60}")

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)

            optimizer.zero_grad()

            # 混合精度前向传播
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # 统计信息
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().tolist())
            train_trues.extend(labels.cpu().tolist())
            train_probs.append(torch.softmax(outputs, dim=1).cpu().detach())

            # 计算批次准确率
            batch_correct = (preds == labels).sum().item()
            batch_acc = 100. * batch_correct / batch_size

            # 定期打印训练日志
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                processed_samples = (batch_idx + 1) * batch_size
                total_samples = len(train_loader.dataset)
                progress_percent = 100. * (batch_idx + 1) / len(train_loader)
                avg_loss_so_far = total_loss / processed_samples

                print(f'Epoch {epoch} [{processed_samples}/{total_samples} '
                      f'({progress_percent:.0f}%)]\t'
                      f'Loss: {loss.item():.6f}\t'
                      f'Batch Acc: {batch_acc:.2f}%\t'
                      f'Avg Loss: {avg_loss_so_far:.4f}')

        # 计算训练集整体指标
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = 100. * accuracy_score(train_trues, train_preds)
        train_recall = 100. * recall_score(train_trues, train_preds, average='weighted', zero_division=0)
        train_f1 = 100. * f1_score(train_trues, train_preds, average='weighted')

        # 计算训练集top5准确率
        train_probs_tensor = torch.cat(train_probs, dim=0)
        train_top5_acc = calculate_topk_accuracy(train_probs_tensor, torch.tensor(train_trues), k=5)

        train_time = time.time() - epoch_start_time

        # ==================== 验证阶段 ====================
        val_start_time = time.time()
        model.eval()
        val_preds = []
        val_trues = []
        val_probs = []
        val_loss = 0

        print(f"\n🔍 验证阶段...")

        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                batch_size = imgs.size(0)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_trues.extend(labels.cpu().tolist())
                val_probs.append(torch.softmax(outputs, dim=1).cpu())

        # 计算验证集指标
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * accuracy_score(val_trues, val_preds)
        val_recall = 100. * recall_score(val_trues, val_preds, average='weighted', zero_division=0)
        val_f1 = 100. * f1_score(val_trues, val_preds, average='weighted')

        # 计算top5准确率和mAP
        val_probs_tensor = torch.cat(val_probs, dim=0)
        val_outputs_tensor = torch.cat([torch.log(p) for p in val_probs], dim=0)  # 恢复logits用于mAP计算
        val_top5_acc = calculate_topk_accuracy(val_probs_tensor, torch.tensor(val_trues), k=5)
        val_mAP = calculate_mAP(val_outputs_tensor, torch.tensor(val_trues), num_classes)

        val_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time

        # 更新学习率
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']

        # ==================== 打印epoch总结 ====================
        print(f"\n📊 Epoch {epoch} 结果:")
        print(f"训练 - 损失: {train_loss:.4f} | Top1 Acc: {train_acc:.2f}% | Top5 Acc: {train_top5_acc:.2f}%")
        print(f"      召回率: {train_recall:.2f}% | F1: {train_f1:.2f}% | 用时: {train_time:.1f}s")
        print(f"验证 - 损失: {val_loss:.4f} | Top1 Acc: {val_acc:.2f}% | Top5 Acc: {val_top5_acc:.2f}%")
        print(f"      召回率: {val_recall:.2f}% | F1: {val_f1:.2f}% | mAP: {val_mAP:.2f}%")
        print(f"学习率: {current_lr:.6f} → {new_lr:.6f}")
        print(f"Epoch用时: {epoch_time:.1f}s")
        print(f"{'=' * 60}")

        # ==================== 保存最佳模型 ====================
        if val_acc > best_acc:
            best_acc = val_acc

            # 保存模型权重
            torch.save(model.state_dict(), f"{output_dir}/best_efficientnet.pth")

            # 保存验证指标
            val_metrics = {
                'val_acc': val_acc,
                'val_top5_acc': val_top5_acc,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_mAP': val_mAP,
                'epoch': epoch
            }
            torch.save(val_metrics, f"{output_dir}/best_val_metrics.pth")

            print(f"🏆 新最优模型已保存! 验证Top1准确率: {val_acc:.2f}%")

        # 保存最新模型
        torch.save(model.state_dict(), f"{output_dir}/latest_efficientnet.pth")

        # 提前停止检查
        if new_lr < 1e-7:
            print(f"\n⚠️  学习率过低 ({new_lr:.8f})，提前停止训练")
            break

    # ==================== 最终测试 ====================
    print(f"\n{'=' * 60}")
    print("🧪 最终测试...")
    print(f"{'=' * 60}")

    # 加载最佳模型
    if os.path.exists(f"{output_dir}/best_efficientnet.pth"):
        model.load_state_dict(torch.load(f"{output_dir}/best_efficientnet.pth", map_location=device))

    model.eval()

    test_preds = []
    test_trues = []
    test_probs = []
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            test_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().tolist())
            test_trues.extend(labels.cpu().tolist())
            test_probs.append(torch.softmax(outputs, dim=1).cpu())

    # 计算测试集指标
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100. * accuracy_score(test_trues, test_preds)
    test_recall = 100. * recall_score(test_trues, test_preds, average='weighted', zero_division=0)
    test_f1 = 100. * f1_score(test_trues, test_preds, average='weighted')
    test_precision = 100. * precision_score(test_trues, test_preds, average='weighted', zero_division=0)

    # 计算top5准确率和mAP
    test_probs_tensor = torch.cat(test_probs, dim=0)
    test_outputs_tensor = torch.cat([torch.log(p) for p in test_probs], dim=0)
    test_top5_acc = calculate_topk_accuracy(test_probs_tensor, torch.tensor(test_trues), k=5)
    test_mAP = calculate_mAP(test_outputs_tensor, torch.tensor(test_trues), num_classes)

    # 保存测试结果
    with open(f"{output_dir}/test_results.txt", "w") as f:
        f.write(f"=== 模型信息 ===\n")
        f.write(f"模型名称: EfficientNet\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"数据路径: {data_dir}\n")
        f.write(f"总参数量: {total_params:,} ({params_in_m:.2f}M)\n")
        f.write(f"可训练参数量: {trainable_params:,}\n")
        f.write(f"类别数: {num_classes}\n\n")

        f.write(f"=== 测试结果 ===\n")
        f.write(f"测试损失: {test_loss:.4f}\n")
        f.write(f"Top1准确率: {test_acc:.2f}%\n")
        f.write(f"Top5准确率: {test_top5_acc:.2f}%\n")
        f.write(f"精确率(Precision): {test_precision:.2f}%\n")
        f.write(f"召回率(Recall): {test_recall:.2f}%\n")
        f.write(f"F1分数: {test_f1:.2f}%\n")
        f.write(f"mAP: {test_mAP:.2f}%\n\n")

        f.write(f"=== 训练信息 ===\n")
        f.write(f"最佳验证准确率: {best_acc:.2f}%\n")
        f.write(f"总训练时间: {time.time() - start_time:.1f}s\n")
        f.write(f"训练轮数: {num_epochs}\n")

    print(f"\n✅ 训练完成!")
    print(f"\n📊 模型信息:")
    print(f"总参数量: {params_in_m:.2f}M (可训练: {trainable_params / 1_000_000:.2f}M)")
    print(f"类别数: {num_classes}")

    print(f"\n📊 最终测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"Top1准确率: {test_acc:.2f}%")
    print(f"Top5准确率: {test_top5_acc:.2f}%")
    print(f"精确率: {test_precision:.2f}%")
    print(f"召回率: {test_recall:.2f}%")
    print(f"F1分数: {test_f1:.2f}%")
    print(f"mAP: {test_mAP:.2f}%")

    print(f"\n🏆 最佳验证准确率: {best_acc:.2f}%")
    print(f"⏱️  总训练时间: {time.time() - start_time:.1f}秒")
    print(f"📁 结果保存至: {output_dir}/test_results.txt")
    print(f"{'=' * 60}")

    return {
        'params_in_m': params_in_m,
        'test_acc': test_acc,
        'test_top5_acc': float(test_top5_acc),
        'test_recall': test_recall,
        'test_mAP': test_mAP,
        'test_f1': test_f1
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练EfficientNet模型')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['generated', 'real', 'shouxie', 'yinshua'],
                        help='选择数据集: generated(合成数据), real(真实数据), shouxie(古籍数据), yinshua(印刷数据)')

    args = parser.parse_args()

    # 数据集路径映射
    dataset_paths = {
        'generated': '/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/generated_enhanced_6145',
        'real': '/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/shouxie+yinshua',
        'shouxie': '/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/shouxieti',
        'yinshua': '/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/yinshuati'
    }

    data_dir = dataset_paths[args.dataset]

    print(f"\n{'=' * 60}")
    print(f"🚀 开始训练EfficientNet模型")
    print(f"📂 数据集: {args.dataset}")
    print(f"📁 数据路径: {data_dir}")
    print(f"{'=' * 60}\n")

    metrics = train(data_dir, args.dataset)

    print(f"\n📋 训练完成，主要指标:")
    print(f"模型参数量: {metrics['params_in_m']:.2f}M")
    print(f"测试Top1准确率: {metrics['test_acc']:.2f}%")
    print(f"测试Top5准确率: {metrics['test_top5_acc']:.2f}%")
    print(f"测试召回率: {metrics['test_recall']:.2f}%")
    print(f"测试mAP: {metrics['test_mAP']:.2f}%")
    print(f"测试F1分数: {metrics['test_f1']:.2f}%")
