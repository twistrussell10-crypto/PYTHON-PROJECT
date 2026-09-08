import socket
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from pet_classifier.predict import Predictor
from scripts.launch_app import available_port


def test_port_selection_skips_busy_port():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        busy = occupied.getsockname()[1]
        selected = available_port(busy, attempts=5)
        assert selected != busy
        with socket.socket() as check:
            check.bind(("127.0.0.1", selected))


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "outputs" / "baseline" / "best.pt").is_file(),
    reason="演示集成测试需要本地训练模型 outputs/baseline/best.pt；基础测试无需该文件",
)
def test_demo_reuses_prediction_and_supports_search():
    original = Predictor.predict
    calls = []
    def record(self, image, top_k=5):
        calls.append(1)
        return original(self, image, top_k)
    with patch.object(Predictor, "predict", record):
        app = AppTest.from_file("app.py").run(timeout=60)
        assert not app.exception
        assert len(calls) == 1
        app.text_input[0].set_value("比格").run(timeout=30)
        assert not app.exception
        assert len(calls) == 1, "搜索品种不应重新运行同一张图片预测"
        assert len(app.dataframe[-1].value) == 1
        assert app.dataframe[-1].value.iloc[0]["中文名"] == "比格犬"
        app.radio[0].set_value("上传图片").run(timeout=30)
        assert not app.exception
        assert len(calls) == 1
