# Oxford-IIIT Pet 实验报告

## 任务与方法
识别 37 种猫犬品种。使用 ImageNet 预训练的 MobileNetV3-Large，先训练分类头，再微调全模型。
官方 trainval 内按类别分层划分训练/验证集；只根据验证集准确率选择最佳 checkpoint。
测试数据不参与梯度更新或最佳轮次选择。

## 本次实测结果
| 项目 | 数值 |
|---|---:|
| 官方测试图片数 | 3669 |
| Top-1 准确率 | 90.65% |
| Top-5 准确率 | 99.37% |
| Macro F1 | 0.9055 |
| 猫犬物种准确率（取预测品种所属物种） | 99.35% |
| 最佳模型轮次 | 11 |
| 最佳验证准确率 | 92.12% |

## 结果文件
- `training_curves.png`：训练/验证损失与准确率。
- `confusion_matrix.png`、`confusion_matrix.csv`：真实类别为行、预测类别为列。
- `classification_report.json`：各品种 precision、recall、F1 和 support。
- `predictions.csv`：每张官方测试图的预测，可复核。
- `errors.png`：置信度最高的最多 12 张误分类图片（有错误时生成）。
- `metrics.json`：原始指标与 checkpoint SHA-256，用于对应模型和评估结果。

## 局限与后续实验
这是单次固定随机种子的实验，不能代表所有拍摄环境。输出分数是 softmax 分数，未做概率校准。
模型只会在已知 37 个品种中选择；不识别未知品种、混血或非宠物图片，也不具备可靠的拒识能力。
可在后续学习中研究相似品种混淆、概率校准、更强主干和多种子对比；新方案应使用验证集选择。

## 来源
- [Oxford 官方数据集与许可](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Torchvision MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)
