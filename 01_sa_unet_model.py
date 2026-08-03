#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SA-UNet 模型结构，用于加载已有 DRIVE h5 权重。

中文说明：
原项目 `SA-UNet-master` 使用 Keras 2.3 / TensorFlow 1.14。
当前环境是新版 TensorFlow/Keras，因此这里用 `tf.keras` 重建同样结构。

注意：
1. `SA_UNet.h5` 是权重文件，不是完整模型文件；
2. DropBlock 在推理阶段等价于恒等映射，且没有可训练权重；
3. 层名显式使用旧 Keras 的名字，例如 conv2d_1、batch_normalization_1；
4. 加载权重时使用 `by_name=True`，对齐 h5 中的旧层名。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from common import ensure_dir, load_config

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / "_matplotlib_cache"))


def import_tensorflow():
    """延迟导入 TensorFlow，便于在环境不兼容时给出清楚错误。"""
    import tensorflow as tf

    return tf


def spatial_attention_tf(x, tf, name_prefix: str = "spatial_attention"):
    """复现原 Spatial_Attention.py 中的空间注意力模块。"""
    avg_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=3, keepdims=True), name=f"{name_prefix}_avg")(x)
    max_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=3, keepdims=True), name=f"{name_prefix}_max")(x)
    concat = tf.keras.layers.Concatenate(axis=3, name="concatenate_4")([avg_pool, max_pool])
    att = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=7,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
        name="conv2d_8",
    )(concat)
    return tf.keras.layers.Multiply(name=f"{name_prefix}_multiply")([x, att])


def conv_bn_relu(x, filters: int, conv_name: str, bn_name: str, act_name: str, tf):
    """原模型中的 Conv2D + DropBlock + BatchNorm + ReLU。

    推理阶段 DropBlock 不改变特征；由于它没有权重，这里省略不会影响 h5 权重加载。
    """
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation=None, padding="same", name=conv_name)(x)
    x = tf.keras.layers.BatchNormalization(name=bn_name)(x)
    x = tf.keras.layers.Activation("relu", name=act_name)(x)
    return x


def build_sa_unet(input_size=(592, 592, 3), start_neurons: int = 16, lr: float = 1e-3):
    """构建与原 `SA_UNet.py` 中 SA_UNet 对齐的 tf.keras 模型。"""
    tf = import_tensorflow()
    n = int(start_neurons)
    inputs = tf.keras.layers.Input(input_size, name="input_1")

    conv1 = conv_bn_relu(inputs, n, "conv2d_1", "batch_normalization_1", "activation_1", tf)
    conv1 = conv_bn_relu(conv1, n, "conv2d_2", "batch_normalization_2", "activation_2", tf)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(conv1)

    conv2 = conv_bn_relu(pool1, n * 2, "conv2d_3", "batch_normalization_3", "activation_3", tf)
    conv2 = conv_bn_relu(conv2, n * 2, "conv2d_4", "batch_normalization_4", "activation_4", tf)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_2")(conv2)

    conv3 = conv_bn_relu(pool2, n * 4, "conv2d_5", "batch_normalization_5", "activation_5", tf)
    conv3 = conv_bn_relu(conv3, n * 4, "conv2d_6", "batch_normalization_6", "activation_6", tf)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_3")(conv3)

    convm = conv_bn_relu(pool3, n * 8, "conv2d_7", "batch_normalization_7", "activation_7", tf)
    convm = spatial_attention_tf(convm, tf)
    convm = conv_bn_relu(convm, n * 8, "conv2d_9", "batch_normalization_8", "activation_8", tf)

    deconv3 = tf.keras.layers.Conv2DTranspose(n * 4, (3, 3), strides=(2, 2), padding="same", name="conv2d_transpose_1")(convm)
    uconv3 = tf.keras.layers.Concatenate(name="concatenate_1")([deconv3, conv3])
    uconv3 = conv_bn_relu(uconv3, n * 4, "conv2d_10", "batch_normalization_9", "activation_9", tf)
    uconv3 = conv_bn_relu(uconv3, n * 4, "conv2d_11", "batch_normalization_10", "activation_10", tf)

    deconv2 = tf.keras.layers.Conv2DTranspose(n * 2, (3, 3), strides=(2, 2), padding="same", name="conv2d_transpose_2")(uconv3)
    uconv2 = tf.keras.layers.Concatenate(name="concatenate_2")([deconv2, conv2])
    uconv2 = conv_bn_relu(uconv2, n * 2, "conv2d_12", "batch_normalization_11", "activation_11", tf)
    uconv2 = conv_bn_relu(uconv2, n * 2, "conv2d_13", "batch_normalization_12", "activation_12", tf)

    deconv1 = tf.keras.layers.Conv2DTranspose(n, (3, 3), strides=(2, 2), padding="same", name="conv2d_transpose_3")(uconv2)
    uconv1 = tf.keras.layers.Concatenate(name="concatenate_3")([deconv1, conv1])
    uconv1 = conv_bn_relu(uconv1, n, "conv2d_14", "batch_normalization_13", "activation_13", tf)
    uconv1 = conv_bn_relu(uconv1, n, "conv2d_15", "batch_normalization_14", "activation_14", tf)

    logits = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation=None, name="conv2d_16")(uconv1)
    output = tf.keras.layers.Activation("sigmoid", name="activation_15")(logits)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="SA_UNet_DRIVE")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_drive_weights(model, weights_path: str | Path):
    """加载 SA-UNet-master/Model/DRIVE/SA_UNet.h5 权重。"""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"找不到 h5 权重文件: {weights_path}")
    # Keras 3 对 legacy h5 仍支持 by_name 加载；如果失败，异常会被上层写入日志。
    model.load_weights(str(weights_path), by_name=True, skip_mismatch=False)
    return model


def main() -> None:
    """检查模型能否构建并加载 h5 权重。"""
    cfg = load_config()
    out_dir = ensure_dir(Path(cfg["output_dir"]) / "01_model_check")
    rows = []
    try:
        model = build_sa_unet(
            input_size=(int(cfg["sa_unet"]["image_size"]), int(cfg["sa_unet"]["image_size"]), 3),
            start_neurons=int(cfg["sa_unet"]["start_neurons"]),
        )
        load_drive_weights(model, cfg["sa_unet"]["weights_path"])
        rows.append({
            "status": "loaded",
            "weights_path": str(cfg["sa_unet"]["weights_path"]),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "parameters": int(model.count_params()),
        })
    except Exception as exc:
        rows.append({"status": "load_failed", "weights_path": str(cfg["sa_unet"]["weights_path"]), "error": repr(exc)})
    pd.DataFrame(rows).to_csv(out_dir / "model_check.csv", index=False)
    print(f"模型检查完成: {out_dir / 'model_check.csv'}")


if __name__ == "__main__":
    main()
