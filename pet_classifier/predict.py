"""训练、测试和上传预测共用同一套图像预处理。"""
import torch
from PIL import Image, ImageOps

from .data import image_transform
from .model import load_checkpoint, select_device


class Predictor:
    """封装模型加载和单张图片推理，供网页与脚本共同调用。"""

    def __init__(self, checkpoint="outputs/baseline/best.pt", device="auto"):
        """加载一次模型并准备固定的验证/预测图像变换。"""
        self.device = select_device(device)
        torch.set_num_threads(4)
        self.model, self.checkpoint = load_checkpoint(checkpoint, self.device)
        self.transform = image_transform()

    @torch.inference_mode()
    def predict(self, image, top_k=5):
        """预测一张 PIL 图片，返回按分数从高到低排列的前 K 个候选。

        每个候选包含类别编号、英文品种名、物种和 softmax 分数。这里的分数是
        37 个已知类别之间的相对值，没有经过概率校准。
        """
        if not 1 <= top_k <= 37:
            raise ValueError("top_k 必须在 1 到 37 之间")
        # 手机照片可能把旋转方向存在 EXIF 中；先校正方向，再统一成三通道 RGB。
        image = ImageOps.exif_transpose(image).convert("RGB")
        # 模型要求批次输入，[3,224,224] 增加批次维后变成 [1,3,224,224]。
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        # softmax 在类别维上把 logits 转为总和为 1 的分数。
        probabilities = self.model(tensor).softmax(1)[0]
        values, indices = probabilities.topk(top_k)
        result = []
        for value, index in zip(values.tolist(), indices.tolist()):
            entry = self.checkpoint["classes"][index]
            result.append({"class_id": index, "breed": entry["name"],
                           "species": entry["species"], "probability": value})
        return result

    def predict_file(self, path, top_k=5):
        """从图片路径读取文件，再调用 predict；with 会在预测后关闭文件。"""
        with Image.open(path) as image:
            return self.predict(image, top_k)
