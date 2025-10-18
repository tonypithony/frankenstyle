# source TF21GPU/bin/activate

# pip install tensorflow[and-cuda] tensorrt keras opencv-python

# https://github.com/keras-team/keras-io/blob/master/examples/generative/neural_style_transfer.py
# https://keras.io/examples/generative/neural_style_transfer/

# conda activate iTF2

# https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5


import tensorrt
from glob import glob
# import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
# from tensorflow.python.client import device_lib 
# print(device_lib.list_local_devices())
# import cv2 
import keras
import numpy as np
import tensorflow as tf
from keras.applications import vgg19#,
                                #mobilenet,
                                #vgg16) # https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

# # Конфигурация доступных GPU так, чтобы память на них выделялась участками в зависимости от потребностей
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)


# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# # Так, скажем, выделяем TensorFlow 40% всего объма видеопамяти
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
# config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# session = tf.compat.v1.Session(config=config)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.86)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

model = vgg19.VGG19(weights="imagenet", include_top=False)
# print(model.summary())
# print([layer.name for layer in model.layers])
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

style_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

content_layer_name = "block5_conv2"


def image_size(base_image_path):
    width, height = keras.utils.load_img(base_image_path).size
    img_nrows = 600
    img_ncols = int(width * img_nrows / height)    
    return [img_nrows, img_ncols]


def preprocess_image(image_path):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.utils.load_img(image_path, target_size=(image_size(image)[0], image_size(image)[1]))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # Util function to convert a tensor into a valid image
    x = x.reshape((image_size(image)[0], image_size(image)[1], 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = image_size(image)[0] * image_size(image)[1]
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


def total_variation_loss(x):
    a = tf.square(
        x[:, : image_size(image)[0] - 1, : image_size(image)[1] - 1, :] - x[:, 1:, : image_size(image)[1] - 1, :]
    )
    b = tf.square(
        x[:, : image_size(image)[0] - 1, : image_size(image)[1] - 1, :] - x[:, : image_size(image)[0] - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())
    
    # Weights of the different loss components
    total_variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


def training_loop(base_image, style_reference_image, iterations):
    base_image_path = base_image # "https://i.imgur.com/F28w3Ac.jpg")
    style_reference_image_path = style_reference_image # "https://i.imgur.com/9ooB60I.jpg")
    iterations = iterations

    result_prefix = f"/media/pendrive/styleclasses/{style_reference_image_path[:-4]}/{base_image[-5:-4]}_vgg19_{iterations}"

    optimizer = keras.optimizers.SGD(
        keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )

    base_image = preprocess_image(base_image_path)
    style_reference_image = preprocess_image(style_reference_image_path)
    combination_image = tf.Variable(preprocess_image(base_image_path))

    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(
            combination_image, base_image, style_reference_image
        )
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy())
            fname = result_prefix  + f'_{style_reference_image_path[:-4]}' + "_at_iteration_%d.png" % i
            keras.utils.save_img(fname, img)
    

if __name__ == "__main__":
    images_path = glob('/home/van_rossum/Изображения/*.jpg')
    for image in images_path:
        training_loop(base_image=image, style_reference_image='1.jpg', iterations=2000)
        training_loop(base_image=image, style_reference_image='2.jpg', iterations=2000)
        training_loop(base_image=image, style_reference_image='3.jpg', iterations=2000)
        training_loop(base_image=image, style_reference_image='4.jpg', iterations=2000)
        training_loop(base_image=image, style_reference_image='5.jpg', iterations=2000)
