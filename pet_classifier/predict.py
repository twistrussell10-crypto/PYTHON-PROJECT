"""训练、测试和上传预测共用同一套图像预处理。"""
import torch
from PIL import Image, ImageOps

from .data import image_transform
from .model import load_checkpoint, select_device


class Predictor:
    def __init__(self, checkpoint="outputs/baseline/best.pt", device="auto"):
        self.device = select_device(device)
        torch.set_num_threads(4)
        self.model, self.checkpoint = load_checkpoint(checkpoint, self.device)
        self.transform = image_transform()

    @torch.inference_mode()
    def predict(self, image, top_k=5):
        if not 1 <= top_k <= 37:
            raise ValueError("top_k 必须在 1 到 37 之间")
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probabilities = self.model(tensor).softmax(1)[0]
        values, indices = probabilities.topk(top_k)
        result = []
        for value, index in zip(values.tolist(), indices.tolist()):
            entry = self.checkpoint["classes"][index]
            result.append({"class_id": index, "breed": entry["name"],
                           "species": entry["species"], "probability": value})
        return result

    def predict_file(self, path, top_k=5):
        with Image.open(path) as image:
            return self.predict(image, top_k)
