# Oxford-IIIT 宠物品种分类器

一个可本地运行、便于逐步学习的 PyTorch 图像分类项目：识别 Oxford-IIIT Pet 数据集中的 **37 个猫犬品种**。

包含官方数据下载、固定种子的分层划分、两阶段迁移学习、最佳模型保存、独立测试集评估、图片预测，以及中文 Streamlit 演示界面。

## 先使用

这台电脑的 `.venv` 已复用现有 PyTorch CUDA 环境，额外依赖安装在项目虚拟环境中。

**双击 `start_app.bat`**，在打开的浏览器中上传照片或选择示例图片即可。

界面的「实验结果」标签页显示实际评估指标。最佳模型保存在 `outputs/baseline/best.pt`，报告在 `outputs/baseline/evaluation/REPORT.md`。

也可以在 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pet_classifier predict examples/Abyssinian.jpg
```

命令行默认使用可用的 GPU；演示界面使用 CPU，方便在没有独立显卡的电脑上预测。

## 在新电脑安装

建议 Python 3.9–3.12。双击 `setup.bat`，或在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如需 GPU，先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配的 PyTorch / Torchvision 与 CUDA 版本，再安装其余依赖。本项目验证环境为 PyTorch 2.6.0+cu124 / Torchvision 0.21.0+cu124。

数据和权重较大，不进入 Git；复制项目到新电脑时，另行复制 `outputs/baseline/best.pt`，或者重新训练。

本次图片通过 [Zenodo 数据集镜像](https://zenodo.org/records/8067751) 下载并校验 ZIP 的 MD5，官方标注单独从 Oxford 下载并校验 MD5。镜像中的文件夹划分被忽略，训练与评估始终使用官方标注中的图片名单。数据来源和图片内容指纹记录在 `data/oxford-iiit-pet/provenance.json`，并随训练配置保存。

## 从头训练与评估

```powershell
# 下载约 800 MB 的官方压缩包，解压还需要额外磁盘空间
.\.venv\Scripts\python.exe -m pet_classifier prepare

# 新实验必须使用新输出目录，防止覆盖交付模型
.\.venv\Scripts\python.exe -m pet_classifier train --output outputs/my_run --epochs 12 --freeze-epochs 2 --batch-size 32

# 完成训练后，仅在官方 test 划分上评估
.\.venv\Scripts\python.exe -m pet_classifier evaluate --checkpoint outputs/my_run/best.pt --output outputs/my_run/evaluation

# 对自己的图片预测
.\.venv\Scripts\python.exe -m pet_classifier predict "C:\path\pet.jpg" --checkpoint outputs/my_run/best.pt --top-k 5

# 自动化测试，不下载数据或预训练权重
.\.venv\Scripts\python.exe -m pytest -q
```

若官方整包下载较慢，可运行 `.\.venv\Scripts\python.exe scripts/download_mirror.py`：脚本下载并校验镜像、还原图片目录，再安装官方标注。网络中断后重跑会复用完整分块。

也可双击 `train_model.bat` 生成 `outputs/my_experiment`。再次训练时修改脚本中的 `RUN_DIR`，或使用上方命令指定新目录。演示界面默认读取交付模型；研究新模型时可修改 `app.py` 的 `RUN` 路径。

默认超参数：12 轮，前 2 轮冻结特征提取网络；分类头初始学习率 0.001；微调阶段主干 0.0001、分类头 0.0003；AdamW；余弦学习率；batch size 32；seed 42；验证比例 20%。

训练使用随机裁剪、水平翻转和轻度颜色扰动；验证、测试和预测统一使用预训练权重对应的 232 像素缩放、224 中心裁剪和 ImageNet 标准化。

## 如何避免数据泄漏

- 沿用官方 `trainval` 与 `test`，不重新混合所有图片。
- 仅在 `trainval` 内按类别划分 80% 训练、20% 验证，并保存到 `split.json`。
- 训练时只根据验证准确率选择 `best.pt`，不加载测试集。
- 用选好的 checkpoint 一次性评估完整官方测试集；后续调参也应基于验证集。

固定随机种子和确定性 cuDNN 设置可减少波动，但不同硬件、库版本或算子不保证逐位相同。

## 文件导览

| 文件 | 用途 |
|---|---|
| `pet_classifier/data.py` | 官方标注解析、37 类顺序、数据划分和预处理 |
| `pet_classifier/model.py` | 模型结构和 checkpoint 加载 |
| `pet_classifier/engine.py` | 冻结/微调、训练循环和最佳模型保存 |
| `pet_classifier/evaluate.py` | Top-1、Top-5、Macro F1、各类指标、混淆矩阵和误分类示例 |
| `pet_classifier/predict.py` | 单图预测，返回前 K 类及 softmax 分数 |
| `pet_classifier/__main__.py` | 四个命令的参数入口 |
| `app.py` | 上传照片、本地示例和实验结果界面 |
| `docs/LEARNING_GUIDE.md` | 后续学习路线和关键概念 |
| `notebooks/01_understand_the_project.ipynb` | 交互学习：读报告、预测、检查数据划分和观察错误 |
| `tests/test_project.py` | 数据泄漏、标签映射、灰度输入和冻结行为验证 |

## 输出说明

每次实验保存 `config.json`、`classes.json`、`split.json`、`history.csv`、`best.pt`。评估生成 `metrics.json`、`classification_report.json`、`predictions.csv`、混淆矩阵 CSV/PNG、训练曲线、误分类图和中文报告。

`best.pt` 包含完整模型权重、类别映射、最佳轮次、预处理标识与配置，预测时不再下载 ImageNet 权重。它是用于推理的最佳模型，不包含断点续训需要的优化器状态；中断后可以保留最佳模型并在新目录重新训练。

## 常见问题

- **显存不足**：设置 `--batch-size 16` 或 `8`。
- **Windows 数据加载失败或启动慢**：设置 `--workers 0`。
- **没有 CUDA**：设置 `--device cpu`，训练会更慢。
- **下载失败**：检查网络/代理并重试；Torchvision 会按官方 MD5 校验压缩包。若系统把本地 HTTP 代理误写成 HTTPS，可仅在当前终端把 `HTTPS_PROXY` 改为实际的 HTTP 代理 URL，再运行命令；不要关闭证书校验。
- **预测很有信心却错误**：softmax 未校准，不等于真实可靠率；数据集以外的动物或物品也会被强制分到 37 类之一。

## 数据来源与使用范围

[Oxford-IIIT Pet 官方数据集](https://www.robots.ox.ac.uk/~vgg/data/pets/) 由 Parkhi、Vedaldi、Zisserman、Jawahar 发布，包含 37 类、每类约 200 张图像。数据遵循官方页面列出的 CC BY-SA 4.0，原图片版权归各自所有者。`examples` 和报告里的图片来自该数据集，不是本项目创作。

参考：[Cats and Dogs, CVPR 2012](https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/)、[Torchvision OxfordIIITPet](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html)、[MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)。

本项目用于学习和品种识别演示，也是一个用于完成暑期 Python 课程的深度学习项目。未实现目标检测、多宠物分别识别、未知类别拒识或混血判断。
