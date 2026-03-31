import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from torch.cuda.amp import autocast
from model import DenseNetModel
from dataset import get_loaders
import time
import argparse


def calculate_topk_accuracy(outputs, labels, k=5):
    """计算top-k准确率"""
    _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
    correct_k = correct[:, :k].reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / labels.size(0))


def test(data_dir, dataset_name, model_path=None, output_dir=None):
    """
    测试已训练的DenseNet-121模型

    Args:
        data_dir: 数据集路径
        dataset_name: 数据集名称
        model_path: 模型权重文件路径（可选，默认自动查找）
        output_dir: 输出目录（可选，默认自动生成）
    """

    # 自动生成路径
    if output_dir is None:
        output_dir = f"outputs-{dataset_name}-0228"

    if model_path is None:
        model_path = f"{output_dir}/best_densenet.pth"

    print(f"\n{'=' * 60}")
    print("🧪 开始测试DenseNet-121...")
    print(f"📂 数据集: {dataset_name}")
    print(f"📁 数据路径: {data_dir}")
    print(f"{'=' * 60}")

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 加载数据
    print("📥 加载数据...")
    train_loader, val_loader, test_loader, class_names = get_loaders(data_dir)
    num_classes = len(class_names)

    print(f"📊 数据集信息: 测试集={len(test_loader.dataset)} 类别数={num_classes}")

    # 模型初始化
    print("📥 加载模型...")
    model = DenseNetModel(num_classes=num_classes, pretrained=False).to(device)

    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ 模型加载成功: {model_path}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_in_m = total_params / 1_000_000

    print(f"📈 模型参数量: {params_in_m:.2f}M (可训练: {trainable_params / 1_000_000:.2f}M)")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 混合精度设置
    use_amp = str(device) == 'cuda'

    # 测试
    model.eval()
    test_preds = []
    test_trues = []
    test_probs = []
    test_loss = 0

    print("\n🔍 推理中...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)

            if use_amp:
                with autocast():
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

            # 显示进度
            if (batch_idx + 1) % 10 == 0:
                print(f"  处理进度: {batch_idx + 1}/{len(test_loader)} batches")

    inference_time = time.time() - start_time

    # 计算测试集指标
    print("\n📊 计算指标...")
    test_loss = test_loss / len(test_loader.dataset)

    # ====== 单标签多分类：Top1（整体准确率）======
    test_acc = 100. * accuracy_score(test_trues, test_preds)

    # ====== weighted：按每类样本数加权（高频类影响更大）======
    weighted_precision = 100. * precision_score(
        test_trues, test_preds, average='weighted', zero_division=0
    )
    weighted_recall = 100. * recall_score(
        test_trues, test_preds, average='weighted', zero_division=0
    )
    weighted_f1 = 100. * f1_score(
        test_trues, test_preds, average='weighted', zero_division=0
    )

    # ====== macro：各类别一视同仁（更能反映长尾/低频类表现）======
    macro_precision = 100. * precision_score(
        test_trues, test_preds, average='macro', zero_division=0
    )
    macro_recall = 100. * recall_score(
        test_trues, test_preds, average='macro', zero_division=0
    )
    macro_f1 = 100. * f1_score(
        test_trues, test_preds, average='macro', zero_division=0
    )

    # ====== micro：按总体 TP/FP/FN 计算（多分类时 micro-F1≈accuracy）======
    micro_precision = 100. * precision_score(
        test_trues, test_preds, average='micro', zero_division=0
    )
    micro_recall = 100. * recall_score(
        test_trues, test_preds, average='micro', zero_division=0
    )
    micro_f1 = 100. * f1_score(
        test_trues, test_preds, average='micro', zero_division=0
    )

    # 计算top5准确率
    test_probs_tensor = torch.cat(test_probs, dim=0)
    test_top5_acc = calculate_topk_accuracy(test_probs_tensor, torch.tensor(test_trues), k=5)

    # 保存测试结果
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "test_results.txt")

    with open(results_file, "w") as f:
        f.write(f"=== 模型信息 ===\n")
        f.write(f"模型名称: DenseNet-121\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"数据路径: {data_dir}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"总参数量: {total_params:,} ({params_in_m:.2f}M)\n")
        f.write(f"可训练参数量: {trainable_params:,}\n")
        f.write(f"类别数: {num_classes}\n\n")

        f.write(f"=== 测试结果 ===\n")
        f.write(f"测试损失: {test_loss:.4f}\n")
        f.write(f"Top1准确率: {test_acc:.2f}%\n")
        f.write(f"Top5准确率: {test_top5_acc:.2f}%\n\n")

        f.write(f"Weighted Precision: {weighted_precision:.2f}%\n")
        f.write(f"Weighted Recall: {weighted_recall:.2f}%\n")
        f.write(f"Weighted F1: {weighted_f1:.2f}%\n\n")

        f.write(f"Macro Precision: {macro_precision:.2f}%\n")
        f.write(f"Macro Recall: {macro_recall:.2f}%\n")
        f.write(f"Macro F1: {macro_f1:.2f}%\n\n")

        f.write(f"Micro Precision: {micro_precision:.2f}%\n")
        f.write(f"Micro Recall: {micro_recall:.2f}%\n")
        f.write(f"Micro F1: {micro_f1:.2f}%\n\n")

        f.write(f"=== 测试信息 ===\n")
        f.write(f"测试样本数: {len(test_loader.dataset)}\n")
        f.write(f"推理时间: {inference_time:.2f}s\n")
        f.write(f"平均推理速度: {len(test_loader.dataset)/inference_time:.2f} samples/s\n")

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"✅ 测试完成!")
    print(f"{'=' * 60}")
    print(f"\n📊 模型信息:")
    print(f"模型: DenseNet-121")
    print(f"数据集: {dataset_name}")
    print(f"总参数量: {params_in_m:.2f}M (可训练: {trainable_params / 1_000_000:.2f}M)")
    print(f"类别数: {num_classes}")

    print(f"\n📊 测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    print(f"Top1准确率: {test_acc:.2f}%")
    print(f"Top5准确率: {test_top5_acc:.2f}%")
    print(f"\nWeighted Precision: {weighted_precision:.2f}%")
    print(f"Weighted Recall: {weighted_recall:.2f}%")
    print(f"Weighted F1: {weighted_f1:.2f}%")
    print(f"\nMacro Precision: {macro_precision:.2f}%")
    print(f"Macro Recall: {macro_recall:.2f}%")
    print(f"Macro F1: {macro_f1:.2f}%")
    print(f"\nMicro Precision: {micro_precision:.2f}%")
    print(f"Micro Recall: {micro_recall:.2f}%")
    print(f"Micro F1: {micro_f1:.2f}%")

    print(f"\n⏱️  推理时间: {inference_time:.2f}秒")
    print(f"⚡ 平均推理速度: {len(test_loader.dataset)/inference_time:.2f} samples/s")
    print(f"📁 结果保存至: {results_file}")
    print(f"{'=' * 60}")

    return {
        'params_in_m': params_in_m,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_top5_acc': float(test_top5_acc),
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'inference_time': inference_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试DenseNet-121模型')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['generated', 'real', 'shouxie', 'yinshua'],
                        help='选择数据集: generated(合成数据), real(真实数据), shouxie(古籍数据), yinshua(印刷数据)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='模型权重路径（可选）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（可选）')

    args = parser.parse_args()

    # 数据集路径映射
    dataset_paths = {
        'generated': '/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/generated_enhanced_6145',
        'real': '/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/shouxie+yinshua',
        'shouxie': '/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/shouxieti',
        'yinshua': '/disks/sdb/user_space/siyuanxue/DL-xixia/dataset/yinshuati'
    }

    data_dir = dataset_paths[args.dataset]

    metrics = test(data_dir, args.dataset, args.model_path, args.output_dir)

    print(f"\n📋 测试完成，主要指标:")
    print(f"数据集: {args.dataset}")
    print(f"模型参数量: {metrics['params_in_m']:.2f}M")
    print(f"测试Top1准确率: {metrics['test_acc']:.2f}%")
    print(f"测试Top5准确率: {metrics['test_top5_acc']:.2f}%")
    print(f"\nWeighted F1: {metrics['weighted_f1']:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.2f}%")
    print(f"Micro F1: {metrics['micro_f1']:.2f}%")
