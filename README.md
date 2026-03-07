# 基于 U-Net 的眼底视网膜血管分割研究

本仓库已初始化为一个可直接扩展的 PyTorch 分割项目，支持：
- `U-Net`
- `Attention U-Net`
- 复合损失：`DiceLoss + FocalLoss`
- 训练/验证指标：`Dice`、`IoU`
- 推理保存二值分割图

## 1. 项目目录

```text
project_root/
├── configs/              # 配置文件 (yaml)
├── data/                 # 数据集存放
├── src/
│   ├── datasets.py       # Dataset
│   ├── models.py         # U-Net, Attention U-Net
│   ├── losses.py         # DiceLoss, FocalLoss, CombinedLoss
│   ├── train.py          # 训练逻辑
│   ├── predict.py        # 推理逻辑
│   └── utils.py          # 指标与工具函数
├── experiments/          # 日志与权重输出
├── requirements.txt
└── main.py               # 入口
```

## 2. 数据准备建议（Kaggle）

请按下面结构整理数据（文件名一一对应）：

```text
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    └── images/
```

说明：
- `masks` 为单通道标注图，前景（血管）像素值建议为 255，背景为 0。
- 如果你目前只有 train/test，可先从 train 划分 10%-20% 作为 val。

## 3. 快速启动

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 训练

```bash
python main.py --mode train --config configs/base.yaml
```

3) 推理

```bash
python main.py --mode predict --config configs/base.yaml
```

## 4. 你的学习路线（建议 6 阶段）

1. 基线阶段（1-2 周）
- 跑通标准 U-Net + Dice/BCE。
- 在验证集稳定复现一个可报告基线（Dice、IoU、AUC）。

2. 数据理解阶段
- 统计血管像素占比（类别不平衡程度）。
- 检查图像分辨率、光照、病变样本比例。

3. 模型改进阶段
- Attention U-Net、Residual U-Net、轻量编码器替换（如 MobileNetV3 encoder）。
- 多尺度输入或深监督。

4. 损失与训练策略阶段
- Dice + Focal / Tversky / Boundary loss 组合对比。
- 学习率策略、混合精度、梯度裁剪、EMA。

5. 泛化与鲁棒性阶段
- 跨数据集测试（如 DRIVE 训练 -> CHASEDB1 测试）。
- TTA、阈值优化、后处理（连通域、形态学细化）。

6. 论文与展示阶段
- 消融实验：每个改动带来多少提升。
- 可视化：错误热区、细小血管召回对比图。

## 5. 创新点与亮点（你可以选 1-2 个主线）

### 创新点 A：边界增强与细血管感知
- 在损失中加入边界约束（如 Boundary loss 或 clDice）。
- 目标：提升细血管召回，减少断裂。

### 创新点 B：频域/对比学习辅助
- 对输入做频域增强（高频结构强化）或自监督预训练。
- 目标：在小样本场景提升泛化。

### 创新点 C：轻量化与临床可部署
- 在不显著损失精度前提下降低参数量与推理时延。
- 目标：做“精度-速度”双指标报告。

## 6. 训练优化重点（优先级从高到低）

1. 数据增强
- 几何变换（flip/rotate/scale）+ 光照增强（CLAHE、gamma）。

2. 采样与类别不平衡
- Patch 训练时提高含血管 patch 采样概率。

3. 损失组合
- 基线：Dice + Focal。
- 进阶：Dice + Tversky 或 Dice + Boundary。

4. 学习率与优化器
- `AdamW + CosineAnnealingLR`（已默认）。

5. 推理后处理
- 小连通域去噪 + skeleton consistency 检查。

## 7. 当前代码下一步建议

- 在 `src/datasets.py` 加入 `albumentations` 增强。
- 在 `src/utils.py` 增加 `AUC / Sensitivity / Specificity` 指标。
- 在 `src/train.py` 增加早停与 TensorBoard 记录。
- 增加 `k-fold` 脚本保证结果可靠性。

## 8. 参考训练命令（建议）

```bash
python main.py --mode train --config configs/base.yaml
```

如果你愿意，我下一步可以直接帮你继续做三项增强：
1) 加入 `albumentations` 强化数据增强；
2) 加入 AUC、SE、SP、F1 指标；
3) 加入早停 + TensorBoard + 最佳阈值搜索。
