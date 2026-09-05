"""生成学习笔记本；笔记本读取现有交付结果，不会自动重新训练。"""
import json
from pathlib import Path

cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(True)})


markdown("""# Oxford-IIIT 宠物分类器：从使用到理解

这份笔记本用于项目完成后的学习。选择项目 `.venv` 作为 Python 内核，然后逐格运行。
它读取已经训练好的模型和报告，不会启动新的训练。安装 Jupyter 内核支持时可运行 `python -m pip install ipykernel`。
""")
code("""from pathlib import Path
import sys
import json

ROOT = Path.cwd()
if not (ROOT / 'pet_classifier').is_dir():
    ROOT = ROOT.parent
assert (ROOT / 'pet_classifier').is_dir(), '请从项目根目录或 notebooks 目录打开笔记本'
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'outputs' / 'baseline'
""")
markdown("## 1. 先看这次训练到底得到了什么\n这些数值来自完整官方测试集的真实评估。")
code("""import pandas as pd
metrics = json.loads((RUN / 'evaluation' / 'metrics.json').read_text(encoding='utf-8'))
pd.Series(metrics)
""")
markdown("## 2. 让模型识别一张照片\n观察前五个候选，分数是 softmax 输出，不是经过校准的正确概率。")
code("""from PIL import Image
from IPython.display import display
from pet_classifier.predict import Predictor

image = Image.open(ROOT / 'examples' / 'Abyssinian.jpg').convert('RGB')
display(image.resize((320, round(image.height * 320 / image.width))))
predictor = Predictor(RUN / 'best.pt', device='cpu')
pd.DataFrame(predictor.predict(image))
""")
markdown("## 3. 查看数据划分\n模型只通过训练集更新参数；验证集选最佳轮次；测试集用于最后报告。")
code("""from pet_classifier.data import records
split = json.loads((RUN / 'split.json').read_text(encoding='utf-8'))
test = {r['name'] for r in records(ROOT / 'data', 'test')}
train, validation = set(split['train']), set(split['validation'])
assert not train & validation
assert not (train | validation) & test
{'train': len(train), 'validation': len(validation), 'test': len(test)}
""")
markdown("## 4. 追踪图片与网络的形状\n批次形状为 `[N, 3, 224, 224]`，输出为 `[N, 37]`。交叉熵接收 logits；展示预测时再用 softmax。")
code("""import torch
tensor = predictor.transform(image).unsqueeze(0)
with torch.inference_mode():
    logits = predictor.model(tensor)
    probabilities = logits.softmax(dim=1)
print('输入:', tuple(tensor.shape), '输出:', tuple(logits.shape))
print('37 类分数之和:', probabilities.sum().item())
""")
markdown("## 5. 阅读训练曲线\n准确率越高通常越好；损失越低通常越好。注意训练增强、Dropout 和验证预处理不同，因此验证准确率可能高于训练准确率。")
code("""history = pd.read_csv(RUN / 'history.csv')
display(history)
display(Image.open(RUN / 'evaluation' / 'training_curves.png'))
""")
markdown("## 6. 从错误里学习\n不要只看总准确率。查看哪些品种最容易混淆，观察高分错误。")
code("""predictions = pd.read_csv(RUN / 'evaluation' / 'predictions.csv')
display(predictions.loc[~predictions['correct']].sort_values('probability', ascending=False).head(12))
display(Image.open(RUN / 'evaluation' / 'errors.png'))
""")
markdown("""## 7. 下一步：设计一次对照实验

先阅读 `docs/LEARNING_GUIDE.md`，然后保持 seed 和划分不变，每次只修改一个变量。

例如在终端运行：
```bash
python -m pet_classifier train --output outputs/experiment_8epochs --epochs 8
```

先比较验证集，不要用测试集反复选参数。训练命令会拒绝覆盖已有模型，便于保留每次实验。
""")
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.12"}}, "nbformat": 4, "nbformat_minor": 4}
destination = Path(__file__).resolve().parents[1] / "notebooks" / "01_understand_the_project.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(destination)
