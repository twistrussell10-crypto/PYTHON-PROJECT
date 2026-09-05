import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Oxford-IIIT 宠物品种分类器")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="下载并校验官方数据")
    prepare.add_argument("--data", default="data")
    train = commands.add_parser("train", help="训练并保存最佳验证模型")
    train.add_argument("--data", default="data")
    train.add_argument("--output", default="outputs/baseline")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--freeze-epochs", type=int, default=2)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="在官方测试集评估")
    evaluate.add_argument("--data", default="data")
    evaluate.add_argument("--checkpoint", default="outputs/baseline/best.pt")
    evaluate.add_argument("--output", default="outputs/baseline/evaluation")
    predict = commands.add_parser("predict", help="预测单张图片")
    predict.add_argument("image")
    predict.add_argument("--checkpoint", default="outputs/baseline/best.pt")
    predict.add_argument("--top-k", type=int, default=5)
    for sub in (train, evaluate, predict):
        sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for sub in (train, evaluate):
        sub.add_argument("--batch-size", type=int, default=32)
        sub.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare(args.data)
    elif args.command == "train":
        from .engine import train
        train(args)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args)
    else:
        from .predict import Predictor
        result = Predictor(args.checkpoint, args.device).predict_file(args.image, args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
