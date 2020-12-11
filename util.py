import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image


def visualize_model(model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def plot_prediction(input, label_idx, preds, class_names):
    desc = os.linesep.join(
        ["%s: %.2f" % (c, preds[i]) for i, c in enumerate(class_names)])
    t = plt.text(-10, 30, desc,
                 horizontalalignment='right',
                 verticalalignment='center')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))
    imshow(input)

    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    plt.title(class_names[label_idx])
    plt.show()


def show_activation(img):
    """
    Convenience wrapper around plt.imshow
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 2:
        plt.imshow(img, cmap=plt.get_cmap('jet'), interpolation='bicubic')
    else:
        plt.imshow(img, interpolation='bicubic')
    plt.axis('off')


def imshow(inp, title=None):
    """Imshow for Tensor.
    taken from
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)