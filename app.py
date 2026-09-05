"""本地图片上传演示：运行 python -m streamlit run app.py。"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from pet_classifier.predict import Predictor

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs" / "baseline"
st.set_page_config(page_title="Pet Atlas · 宠物品种识别", page_icon="🐾", layout="wide")


@st.cache_resource
def get_predictor(path, modified):
    return Predictor(path, device="cpu")


st.caption("PET ATLAS / OXFORD-IIIT PET")
st.title("一张照片，认识它的品种。")
st.write("上传猫咪或狗狗的照片，查看模型预测的品种与前五名候选。")
with st.sidebar:
    st.header("关于这个模型")
    st.write("MobileNetV3-Large · 37 个猫犬品种")
    st.caption("在本机运行，上传图片不写入项目磁盘。")
    st.info("仅支持数据集中的 37 个品种。非宠物、混血或其他品种也可能得到高分；分数不等于可靠性保证。")
    st.markdown("[查看数据集来源](https://www.robots.ox.ac.uk/~vgg/data/pets/)")
    st.caption("图片数据遵循原始许可，版权属于各图片所有者。")

checkpoint = RUN / "best.pt"
if not checkpoint.exists():
    st.warning("尚未找到已训练模型。请在项目目录运行下面两条命令。")
    st.code("python -m pet_classifier prepare\npython -m pet_classifier train", language="bash")
    st.stop()

tab_predict, tab_report, tab_classes = st.tabs(["图片识别", "实验结果", "支持的品种"])
with tab_predict:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        upload = st.file_uploader("选择宠物照片", type=["jpg", "jpeg", "png", "webp"])
        examples = sorted((ROOT / "examples").glob("*.jpg"))
        selected = st.selectbox("或者使用示例图片", ["请选择"] + [p.stem for p in examples])
        source = upload if upload is not None else next((p for p in examples if p.stem == selected), None)
        image = None
        if source is not None:
            try:
                with Image.open(source) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                st.image(image, width="stretch")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                st.error("无法读取这张图片，请换一张有效的 JPG、PNG 或 WebP 图片。")
    with right:
        if image is None:
            st.subheader("等待一位毛茸茸的访客")
            st.write("选择图片后会自动开始识别。建议使用主体清楚、只有一只宠物的照片。")
        else:
            with st.spinner("正在观察照片…"):
                predictor = get_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns)
                results = predictor.predict(image)
            best = results[0]
            species = "猫" if best["species"] == "cat" else "狗"
            st.caption(f"预测结果 · {species}")
            st.subheader(best["breed"].replace("_", " ").title())
            st.metric("模型分数", f"{best['probability']:.1%}")
            for result in results:
                st.write(f"{result['breed'].replace('_', ' ').title()} · {result['probability']:.1%}")
                st.progress(result["probability"])
            if best["probability"] < 0.5:
                st.warning("候选分数较分散，可以尝试更清晰的正面照片。")
            st.caption("这些是 37 类 softmax 分数；前五名的分数之和可能小于 100%。")
            st.download_button("下载预测 JSON", json.dumps(results, ensure_ascii=False, indent=2),
                               file_name="prediction.json", mime="application/json")

with tab_report:
    evaluation = RUN / "evaluation"
    if (evaluation / "metrics.json").exists():
        metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
        a, b, c = st.columns(3)
        a.metric("测试 Top-1", f"{metrics['top1_accuracy']:.2%}")
        b.metric("测试 Top-5", f"{metrics['top5_accuracy']:.2%}")
        c.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        st.caption(f"官方测试集 {metrics['samples']} 张图片 · 最佳模型第 {metrics['checkpoint_epoch']} 轮")
        for filename, caption in [("training_curves.png", "训练与验证曲线"),
                                  ("confusion_matrix.png", "混淆矩阵"), ("errors.png", "高分误分类示例")]:
            if (evaluation / filename).exists():
                st.image(str(evaluation / filename), caption=caption, width="stretch")
    else:
        st.info("评估完成后，此处将显示真实测试指标。")

with tab_classes:
    predictor = get_predictor(str(checkpoint), checkpoint.stat().st_mtime_ns)
    st.dataframe(pd.DataFrame([{"品种": c["name"].replace("_", " "),
                                "物种": "猫" if c["species"] == "cat" else "狗"}
                               for c in predictor.checkpoint["classes"]]), hide_index=True,
                 width="stretch")
