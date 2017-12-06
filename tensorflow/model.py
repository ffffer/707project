import numpy as np
from PIL import Image

from keras import backend
from keras.models import  Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

image_height = 512
image_width = 512
epoch = 10
content_weight = 0.0025
style_weight = 10.0
total_variation_weight = 0.1
layer_name = 'block1_conv2'


def total_loss(x):
    a = backend.square(x[:, :image_height-1, :image_width-1, :] - x[:, 1:, :image_width-1, :])
    b = backend.square(x[:, :image_height-1, :image_width-1, :] - x[:, :image_height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


def style_loss(style, input):
    style_gram = gram_matrix(style)
    input_gram = gram_matrix(input)
    channels = 3
    size = image_height * image_width
    return backend.sum(backend.square(style_gram - input_gram)) / (4. * (channels ** 2) * (size ** 2))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def content_loss(content, input):
    return backend.sum(backend.square(input - content))


def process_image(file):
    image = Image.open(file)
    image = image.resize((image_width, image_height))
    image_array = np.asarray(image, dtype='float32')
    image_array = np.expand_dims(image_array, axis=0)

    image_array[:, :, :, 0] -= 103.939
    image_array[:, :, :, 1] -= 116.779
    image_array[:, :, :, 2] -= 123.68
    image_array = image_array[:, :, :, ::-1]
    return backend.variable(image_array)


x = backend.placeholder((1, image_height, image_width, 3))
input_tensor = backend.concatenate([process_image("../StarryNight.jpg"), process_image("../mooncake.jpeg"), x], axis=0)

model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
loss = backend.variable(0.0)

layers = dict([(layer.name, layer.output) for layer in model.layers])
layer_features = layers[layer_name]
content_image_features = layer_features[1, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(content_image_features, combination_features)

feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3']
for layer_name in feature_layers:
   layer_features = layers[layer_name]
   style_features = layer_features[0, :, :, :]
   combination_features = layer_features[2, :, :, :]
   sl = style_loss(style_features, combination_features)
   loss += (style_weight / len(feature_layers)) * sl

loss += total_variation_weight * total_loss(x)


grads = backend.gradients(loss, x)

outputs = [loss]
outputs += grads
f_outputs = backend.function([x], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, image_height, image_width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

res_image = np.random.uniform(0, 255, (1, image_height, image_width, 3)) - 128.
for i in range(epoch):
    res_image, min_val, info = fmin_l_bfgs_b(evaluator.loss, res_image.flatten(), fprime=evaluator.grads, maxfun=20)
    print i, min_val


res_image = res_image.reshape((image_height, image_width, 3))
res_image = res_image[:, :, ::-1]
res_image[:, :, 0] += 103.939
res_image[:, :, 1] += 116.779
res_image[:, :, 2] += 123.68
res_image = np.clip(res_image, 0, 255).astype('uint8')

image = Image.fromarray(res_image)
# image.save("style_1.png")
image.save("content/sample_1.png")
