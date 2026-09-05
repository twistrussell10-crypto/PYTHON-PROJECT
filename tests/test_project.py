import copy

import pytest
import torch
from PIL import Image

from pet_classifier.data import PetDataset, stratified_split
from pet_classifier.engine import run_epoch
from pet_classifier.model import build_model
from pet_classifier.predict import Predictor


def test_split_is_stratified_reproducible_and_disjoint():
    rows = [{"name": f"{label}_{i}", "label": label} for label in range(37) for i in range(10)]
    train, val = stratified_split(rows, seed=42)
    assert (train, val) == stratified_split(rows, seed=42)
    assert len(train) == 296 and len(val) == 74
    assert not ({r['name'] for r in train} & {r['name'] for r in val})
    assert {r['name'] for r in train + val} == {r['name'] for r in rows}
    assert all(sum(r['label'] == label for r in val) == 2 for label in range(37))
    assert (train, val) != stratified_split(rows, seed=43)
    with pytest.raises(ValueError):
        stratified_split(rows, val_fraction=0)


def test_checkpoint_prediction_uses_saved_label_order_and_rgb(tmp_path):
    torch.set_num_threads(2)
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.zero_()
        model.classifier[-1].bias[7] = 10
    classes = [{"name": f"breed_{i}", "species": "cat" if i < 12 else "dog"} for i in range(37)]
    path = tmp_path / "model.pt"
    torch.save({"architecture": "mobilenet_v3_large", "model_state": model.state_dict(),
                "classes": classes}, path)
    predictor = Predictor(path, device="cpu")
    for mode in ("RGB", "RGBA", "L"):
        result = predictor.predict(Image.new(mode, (300, 240)), top_k=37)
        assert result[0]['breed'] == 'breed_7'
        assert len(result) == 37
        assert sum(r['probability'] for r in result) == pytest.approx(1, abs=1e-5)
    with pytest.raises(ValueError):
        predictor.predict(Image.new("RGB", (30, 30)), top_k=0)


def test_frozen_backbone_does_not_update_batchnorm_or_weights(tmp_path):
    torch.set_num_threads(2)
    model = build_model(pretrained=False)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    original = copy.deepcopy(model.features.state_dict())
    head_before = model.classifier[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]
    result = run_epoch(model, loader, torch.device("cpu"), optimizer, scaler, frozen=True)
    assert result["loss"] > 0
    assert all(torch.equal(value, model.features.state_dict()[key]) for key, value in original.items())
    assert not torch.equal(head_before, model.classifier[-1].weight)


def test_dataset_handles_grayscale(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (320, 260), color=150).save(path)
    image, label = PetDataset([{"path": str(path), "label": 3}])[0]
    assert image.shape == (3, 224, 224)
    assert label == 3 and torch.isfinite(image).all()
