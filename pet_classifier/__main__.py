"""项目的命令行入口。

当在项目根目录执行 ``python -m pet_classifier`` 时，Python 会先把
``pet_classifier`` 作为包加载，然后寻找并执行这个 ``__main__.py`` 文件。

本文件不负责具体的数据处理、训练或预测，而是完成两件事：

1. 使用 argparse 读取用户在终端输入的命令和选项；
2. 根据用户选择的子命令，把任务交给对应模块中的函数。

四种基本用法如下：

    python -m pet_classifier prepare
    python -m pet_classifier train
    python -m pet_classifier evaluate
    python -m pet_classifier predict examples/Abyssinian.jpg
"""

# argparse 是 Python 标准库中的命令行参数解析工具，不需要额外安装。
import argparse

# json 用于把 predict 命令返回的 Python 列表和字典转换成 JSON 文本。
import json


def main():
    """建立命令行界面，解析用户输入，并调用相应的项目功能。"""

    # ArgumentParser 是整个命令行程序的总解析器。
    # description 会在执行 ``python -m pet_classifier --help`` 时显示。
    parser = argparse.ArgumentParser(description="Oxford-IIIT 宠物品种分类器")

    # add_subparsers 表示程序下面还有多种子命令。
    # dest="command" 表示解析结果将保存在 args.command 中。
    # required=True 表示用户必须选择 prepare、train、evaluate 或 predict 之一。
    commands = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # prepare：下载并准备 Oxford-IIIT Pet 数据集
    # 示例：python -m pet_classifier prepare --data data
    # ------------------------------------------------------------------
    prepare = commands.add_parser("prepare", help="下载并校验官方数据")

    # 以 -- 开头的是可选参数。用户不填写 --data 时，默认使用项目的 data 目录。
    prepare.add_argument("--data", default="data")

    # ------------------------------------------------------------------
    # train：使用训练集更新模型参数，并根据验证集保存最佳模型
    # 示例：python -m pet_classifier train --epochs 12 --device cuda
    # ------------------------------------------------------------------
    train = commands.add_parser("train", help="训练并保存最佳验证模型")

    # 数据目录，其中应包含 oxford-iiit-pet/images 和 annotations。
    train.add_argument("--data", default="data")

    # 训练结果目录，用于保存 best.pt、history.csv、config.json 等文件。
    train.add_argument("--output", default="outputs/baseline")

    # type=int/float 会把终端输入的字符串转换成相应的数值类型。
    # epochs 是总训练轮数；一次完整遍历训练集称为一轮（epoch）。
    train.add_argument("--epochs", type=int, default=12)

    # 前两轮冻结特征提取主干，只训练新替换的 37 类分类头。
    train.add_argument("--freeze-epochs", type=int, default=2)

    # 分类头冻结训练阶段使用的初始学习率。
    train.add_argument("--lr", type=float, default=0.001)

    # 从官方 trainval 中按品种分层抽取 20% 作为验证集。
    train.add_argument("--val-fraction", type=float, default=0.2)

    # 固定随机种子，使数据划分和随机增强尽可能可复现。
    train.add_argument("--seed", type=int, default=42)

    # ------------------------------------------------------------------
    # evaluate：加载已经训练好的模型，在独立官方测试集上计算指标
    # 示例：python -m pet_classifier evaluate --device cpu
    # ------------------------------------------------------------------
    evaluate = commands.add_parser("evaluate", help="在官方测试集评估")
    evaluate.add_argument("--data", default="data")

    # checkpoint 是模型检查点，里面保存网络参数、类别顺序和最佳轮次等信息。
    evaluate.add_argument("--checkpoint", default="outputs/baseline/best.pt")

    # 评估指标、预测 CSV 和分析图片会写入这个目录。
    evaluate.add_argument("--output", default="outputs/baseline/evaluation")

    # ------------------------------------------------------------------
    # predict：使用训练好的模型预测一张图片
    # 示例：python -m pet_classifier predict examples/Abyssinian.jpg --top-k 5
    # ------------------------------------------------------------------
    predict = commands.add_parser("predict", help="预测单张图片")

    # image 没有以 -- 开头，所以是必须提供的“位置参数”。
    # 它在命令中的位置紧跟在 predict 后面。
    predict.add_argument("image")
    predict.add_argument("--checkpoint", default="outputs/baseline/best.pt")

    # top-k 表示输出模型分数最高的前几个候选品种。
    predict.add_argument("--top-k", type=int, default=5)

    # 三个需要模型计算的命令使用相同的设备选择方式。
    # 循环可以避免为三个子命令重复写三次完全相同的 add_argument。
    for sub in (train, evaluate, predict):
        # choices 限制用户只能输入以下三个值：
        # auto 自动选择；cpu 强制使用处理器；cuda 强制使用 NVIDIA 显卡。
        sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    # 训练和测试都以批次读取多张图片，因此共用 batch-size 和 workers 参数。
    for sub in (train, evaluate):
        # batch-size 是一次送入模型的图片数量。
        sub.add_argument("--batch-size", type=int, default=32)

        # workers 是 DataLoader 后台读取图片的子进程数量。
        sub.add_argument("--workers", type=int, default=2)

    # parse_args 真正读取 sys.argv，也就是终端中写在命令后面的内容。
    # 例如输入 ``train --epochs 8`` 后，可通过 args.command 和 args.epochs 读取。
    # argparse 还会自动处理 --help、缺少参数和参数类型错误等情况。
    args = parser.parse_args()

    # 延迟导入使 --help 启动更快，也避免未执行命令加载不必要的依赖。
    # 开头的点表示相对导入，例如 .data 就是 pet_classifier.data。
    if args.command == "prepare":
        from .data import prepare

        # prepare 只需要数据保存目录，所以传入 args.data。
        prepare(args.data)
    elif args.command == "train":
        from .engine import train

        # train 需要多项训练参数，直接接收完整的 argparse Namespace 对象。
        train(args)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        # evaluate 同样从 args 中读取 checkpoint、output、batch_size 等参数。
        evaluate(args)
    else:
        # 由于 command 必须是四个选项之一，前三个分支都不匹配时一定是 predict。
        from .predict import Predictor

        # 第一步加载模型，第二步读取图片并返回前 top_k 个预测结果。
        result = Predictor(args.checkpoint, args.device).predict_file(args.image, args.top_k)

        # ensure_ascii=False 让中文直接显示；indent=2 让 JSON 缩进，便于阅读。
        print(json.dumps(result, ensure_ascii=False, indent=2))


# 直接执行本文件，或使用 ``python -m pet_classifier`` 时，__name__ 等于
# "__main__"，因此调用 main()。如果别的文件只是导入它，则不会自动运行。
if __name__ == "__main__":
    main()
