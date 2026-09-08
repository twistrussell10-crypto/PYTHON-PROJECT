"""本地课程演示：图片识别、真实评估结果和项目原理。"""
import hashlib
import io
import json
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from pet_classifier.labels import breed_zh
from pet_classifier.predict import Predictor

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "outputs" / "baseline"
st.set_page_config(page_title="Pet Atlas · 宠物品种识别", page_icon="🐾", layout="wide")


@st.cache_resource
def get_predictor(path, modified):
    return Predictor(path, device="cpu")


@st.cache_data(show_spinner=False)
def checkpoint_hash(path, modified):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


st.caption("PET ATLAS / OXFORD-IIIT PET")
st.title("认识照片里的宠物品种")
st.write("37 个猫犬品种，支持中文结果、前五名候选和实验报告。")
with st.sidebar:
    st.header("宠物品种分类器")
    st.write("MobileNetV3-Large 迁移学习")
    st.caption("CPU 本地预测，上传图片仅保留在当前会话内存中。")
    st.info("仅覆盖数据集中的 37 个品种。未知品种、混血和非宠物图片也会得到分类结果。")
    st.caption("模型分数未做概率校准，不能当作正确率保证。")
    st.markdown("[项目仓库](https://github.com/twistrussell10-crypto/PYTHON-PROJECT)")
    st.markdown("[Oxford 数据集来源](https://www.robots.ox.ac.uk/~vgg/data/pets/)")

checkpoint = RUN / "best.pt"
if not checkpoint.exists():
    st.warning("尚未找到训练好的模型，请复制 best.pt 到 outputs/baseline，或先训练。")
    st.code("python -m pet_classifier prepare\npython -m pet_classifier train", language="bash")
    st.stop()

modified = checkpoint.stat().st_mtime_ns
try:
    predictor = get_predictor(str(checkpoint), modified)
except (OSError, RuntimeError, ValueError, KeyError) as error:
    st.error(f"模型加载失败：{error}")
    st.stop()

tab_predict, tab_report, tab_flow, tab_classes = st.tabs(["图片识别", "实验结果", "项目原理", "支持的品种"])
with tab_predict:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        mode = st.radio("图片来源", ["示例图片", "上传图片"], horizontal=True)
        payload, example = None, None
        if mode == "上传图片":
            upload = st.file_uploader("上传一张宠物照片", type=["jpg", "jpeg", "png", "webp"])
            if upload is not None:
                payload = upload.getvalue()
        else:
            manifest_path = ROOT / "examples" / "manifest.json"
            manifest = load_json(manifest_path) if manifest_path.exists() else {}
            files = sorted((ROOT / "examples").glob("*.jpg"))
            if files:
                selected = st.selectbox("选择一张示例图片", files,
                    format_func=lambda p: breed_zh(manifest.get(p.name, {}).get("breed", p.stem))
                        + ("（官方测试集）" if manifest.get(p.name, {}).get("split") == "test" else ""))
                payload = selected.read_bytes()
                example = manifest.get(selected.name)
            else:
                st.info("没有示例图片，请切换到上传图片。")
        picture = None
        if payload:
            try:
                with Image.open(io.BytesIO(payload)) as original:
                    picture = ImageOps.exif_transpose(original).convert("RGB")
                st.image(picture, width="stretch")
                if example:
                    st.caption(f"真实标签：{breed_zh(example['breed'])} / {example['breed']}。样例仅用于展示。")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                st.error("无法读取图片，请换一张有效的 JPG、PNG 或 WebP 文件。")
    with right:
        if picture is None:
            st.subheader("选择照片后自动识别")
            st.write("单只宠物、主体清楚的照片更适合这个模型。")
        else:
            key = (hashlib.sha256(payload).hexdigest(), modified)
            if st.session_state.get("prediction_key") != key:
                with st.spinner("正在识别…"):
                    start = time.perf_counter()
                    results = predictor.predict(picture)
                    elapsed = (time.perf_counter() - start) * 1000
                st.session_state.prediction_key = key
                st.session_state.prediction_value = (results, elapsed)
            results, elapsed = st.session_state.prediction_value
            best = results[0]
            st.caption("预测品种所属物种：" + ("猫" if best["species"] == "cat" else "狗"))
            st.subheader(breed_zh(best["breed"]))
            st.write(best["breed"].replace("_", " ").title())
            score, timing = st.columns(2)
            score.metric("模型分数", f"{best['probability']:.1%}")
            timing.metric("本次推理耗时", f"{elapsed:.0f} ms")
            st.caption("耗时包含预处理与前向计算，不含模型加载；重复展示沿用首次结果。")
            for result in results:
                st.write(f"{breed_zh(result['breed'])}　{result['probability']:.1%}")
                st.progress(result["probability"])
            if best["probability"] < 0.5:
                st.warning("候选分数较分散，可以尝试更清晰的正面照片。")
            st.caption("这里只显示前五名，分数之和可能小于 100%。")
            exported = [dict(row, breed_zh=breed_zh(row["breed"])) for row in results]
            st.download_button("下载预测 JSON", json.dumps(exported, ensure_ascii=False, indent=2),
                               file_name="prediction.json", mime="application/json")

with tab_report:
    evaluation = RUN / "evaluation"
    if (evaluation / "metrics.json").exists():
        metrics = load_json(evaluation / "metrics.json")
        if metrics.get("checkpoint_sha256") != checkpoint_hash(str(checkpoint), modified):
            st.warning("这份评估报告与当前模型文件不匹配，请重新运行 evaluate 后再用于汇报。")
        a, b, c = st.columns(3)
        a.metric("测试 Top-1", f"{metrics['top1_accuracy']:.2%}")
        b.metric("测试 Top-5", f"{metrics['top5_accuracy']:.2%}")
        c.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        st.caption(f"官方测试集 {metrics['samples']} 张图片。验证集选择第 {metrics['checkpoint_epoch']} 轮模型。")
        st.write("Top-1 看第一名是否正确，Top-5 看正确品种是否进入前五名。Macro F1 对每个品种等权平均。")
        history_path = RUN / "history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).set_index("epoch")
            h1, h2 = st.columns(2)
            with h1:
                st.subheader("准确率变化")
                st.line_chart(history[["train_accuracy", "val_accuracy"]].rename(
                    columns={"train_accuracy": "训练", "val_accuracy": "验证"}))
            with h2:
                st.subheader("损失变化")
                st.line_chart(history[["train_loss", "val_loss"]].rename(
                    columns={"train_loss": "训练", "val_loss": "验证"}))
        st.caption("训练准确率接近 100%，验证准确率约 92%，说明仍存在泛化差距。")
        report_path = evaluation / "classification_report.json"
        if report_path.exists():
            report = load_json(report_path)
            per_breed = pd.DataFrame([{"品种": breed_zh(name), "精确率": data["precision"],
                "召回率": data["recall"], "F1": data["f1-score"], "测试图片数": int(data["support"])}
                for name, data in report.items() if isinstance(data, dict) and name in
                {item["name"] for item in predictor.checkpoint["classes"]}]).sort_values("F1")
            with st.expander("各品种详细指标，按 F1 从低到高排列"):
                st.dataframe(per_breed, hide_index=True, width="stretch")
        with st.expander("混淆矩阵与高分误分类样例"):
            for filename, caption in [("confusion_matrix.png", "行是真实品种，列是预测品种"),
                                      ("errors.png", "模型也会高分判断错误")]:
                if (evaluation / filename).exists():
                    st.image(str(evaluation / filename), caption=caption, width="stretch")
        if (evaluation / "REPORT.md").exists():
            st.download_button("下载实验报告", (evaluation / "REPORT.md").read_bytes(),
                               file_name="REPORT.md", mime="text/markdown")
    else:
        st.info("评估完成后，此处将显示测试指标。")

with tab_flow:
    st.subheader("训练流程与演示流程")
    st.dataframe(pd.DataFrame([
        ["数据准备", "data.py", "官方 trainval 内划分训练与验证，保留官方 test"],
        ["模型构建", "model.py", "ImageNet 预训练主干，输出层替换为 37 类"],
        ["训练与验证", "engine.py", "训练 12 轮，根据验证准确率保存 best.pt"],
        ["最终评估", "evaluate.py", "完整测试集得到指标、混淆矩阵和逐图预测"],
        ["单图推理", "predict.py", "加载 best.pt，预处理、前向计算、softmax 排序"],
        ["课程演示", "app.py", "展示预测品种、候选分数与实验结果"],
    ], columns=["阶段", "核心文件", "职责"]), hide_index=True, width="stretch")
    st.write("演示时加载训练好的 best.pt，不会再次训练。JSON 保存类别、参数和实验结果。")
    st.code("start_app.bat", language="text")

with tab_classes:
    query = st.text_input("搜索中文或英文品种名")
    species_filter = st.radio("物种筛选", ["全部", "猫", "狗"], horizontal=True)
    classes = pd.DataFrame([{"中文名": breed_zh(c["name"]), "英文类别": c["name"],
        "物种": "猫" if c["species"] == "cat" else "狗"} for c in predictor.checkpoint["classes"]])
    if species_filter != "全部":
        classes = classes[classes["物种"] == species_filter]
    if query:
        mask = classes["中文名"].str.contains(query, case=False, regex=False) | classes["英文类别"].str.contains(query, case=False, regex=False)
        classes = classes[mask]
    st.dataframe(classes, hide_index=True, width="stretch")
