# source TF21GPU/bin/activate

# pip install pydot

import tensorflow as tf
from tensorflow import keras
from keras.applications import vgg19
from tensorflow.keras.utils import plot_model

# Загрузка модели
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Создание feature extractor с несколькими выходами
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
content_layer_name = "block5_conv2"

outputs_dict = {layer.name: layer.output for layer in model.layers}
feature_extractor = keras.Model(inputs=model.input, outputs=outputs_dict)

# Визуализация и сохранение графа
plot_model(
    feature_extractor,
    to_file='feature_extractor.png',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',  # 'TB' = top to bottom, 'LR' = left to right
    expand_nested=False,
    dpi=96
)