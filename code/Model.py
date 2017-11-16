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
lr = 10
EPOCHS = 100


def get_content(model, image):
    return model(image)
#
#
# def get_style(model, image_file):


def content_loss(content, new_image):
    return torch.sqrt(torch.sum(torch.pow(new_image - content, 2)))


def train(model, content):
    content = Variable(content.data, requires_grad=False)
    x = Variable(torch.from_numpy(np.random.uniform(0, 255, (1, 3, 512, 512))).float(), requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1,
                                                           verbose=False, threshold=1e-4, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-8)

    for i in range(EPOCHS):
        model.zero_grad()
        content_x = get_content(model, x)
        l = content_loss(content, content_x)
        l.backward()
        optimizer.step()
        scheduler.step(l.data[0])
        print i, l, optimizer.param_groups[0]['lr']

        if np.isclose(optimizer.param_groups[0]['lr'], 1e-08, rtol=1e-05, atol=1e-08, equal_nan=False):
            break
        if np.isclose(optimizer.param_groups[0]['lr'], 1, rtol=1e-05, atol=1e-08, equal_nan=False):
            scheduler.patience = 20
        if np.isclose(optimizer.param_groups[0]['lr'], 0.1, rtol=1e-05, atol=1e-08, equal_nan=False):
            scheduler.patience = 20

    decoder(x.data.numpy())


def decoder(x):
    reshaped_x = x.reshape(512, 512, 3)
    reshaped_x = reshaped_x[:, :, ::-1]
    reshaped_x[:, :, 0] += 103.939
    reshaped_x[:, :, 1] += 116.779
    reshaped_x[:, :, 2] += 123.68
    image = Image.fromarray(reshaped_x, 'RGB')
    image.show()

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



