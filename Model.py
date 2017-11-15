import torch.nn as nn
import torch
import torchvision.models as models
import vgg16 as vgg_model
from PIL import Image
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter

image_width = 512
image_height = 512


def get_content(model, image):
    return model(image)
#
#
# def get_style(model, image_file):


def content_loss(content, new_image):
    return torch.sum(torch.pow(new_image - content, 2))


def train(model, content):
    x = Variable(torch.from_numpy(np.random.uniform(0, 255, (1, 3, 512, 512))).float(), requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.001)

    for i in range(100):
        optimizer.zero_grad()
        content_x = get_content(model, x)
        l = content_loss(content, content_x)
        l.backward()
        optimizer.step()
        print i, l


def process_image(file):
    image = Image.open(file)
    image = image.resize((image_width, image_height))
    image_array = np.asarray(image, dtype='float32')
    image_array = np.expand_dims(image_array, axis=0)

    image_array[:, :, :, 0] -= 103.939
    image_array[:, :, :, 1] -= 116.779
    image_array[:, :, :, 2] -= 123.68
    image_array = image_array[:, :, :, ::-1]
    image_array = image_array.T.reshape(1, 3, 512, 512)
    return Variable(torch.from_numpy(image_array.copy()))


vgg16 = vgg_model.get_model()
input = process_image("mooncake.jpeg")
train(vgg16, get_content(vgg16, input))



