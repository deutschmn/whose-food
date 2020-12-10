# %%

import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import time
import os
import copy

# %%

# Work around some truncated images: https://stackoverflow.com/a/23575424
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%

# TODO apply separate transforms for validation split
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'data/from'
validation_split = 0.2
random_seed = 132
batch_size = 4

dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

indices = list(range(len(dataset)))
split = int(np.floor(validation_split * len(dataset)))

np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

dataloaders = {
    "train": torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=train_sampler),
    "val": torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       sampler=valid_sampler)
}

dataset_sizes = {x: batch_size * len(dataloaders[x]) for x in ['train', 'val']}

class_names = dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%

# taken from
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


# Get a batch of training data
inputs, classes = next(iter(dataloaders["train"]))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# %%

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# %%

def visualize_model(model, num_images=6):
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


# %%

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7,
                                             gamma=0.1)

# %%

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

# %%

visualize_model(model_ft, num_images=6)


# %%

def plot_prediction(input, label_idx, preds):
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


# %%

# plot some random predictions

top = 10
for i, (inputs, labels) in enumerate(dataloaders["val"]):
    if i == top:
        break
    inputs = inputs.to(device)
    labels = labels.to(device)
    preds = torch.softmax(model_ft(inputs), 1)
    for j, label in enumerate(labels):
        plot_prediction(inputs.cpu().data[j], label, preds[j])

# %%

# Plot some classics

full_val_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler,
                                              batch_size=1)

threshold = 0.99
num_images = 6

for class_name in class_names:
    selected_inputs = []
    selected_preds = []

    class_idx = class_names.index(class_name)
    for i, (inputs, labels) in enumerate(full_val_loader):
        if labels[0] == class_idx:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if len(selected_inputs) == num_images:
                break
            else:
                preds = torch.softmax(model_ft(inputs), 1)

                if preds[0][class_idx] >= threshold:
                    selected_inputs.append(inputs[0])
                    selected_preds.append(preds[0])

    print(f"Classic {class_name}:")
    for i in range(num_images):
        plot_prediction(selected_inputs[i].cpu().data, class_idx,
                        selected_preds[i])
        # print(selected_inputs[i].shape, class_idx, selected_preds[i].shape)

# %%

# Plot some undecideds

full_val_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler,
                                              batch_size=1)

threshold = 0.3
num_images = 6

selected_inputs = []
selected_labels = []
selected_preds = []

for i, (inputs, labels) in enumerate(full_val_loader):
    if len(selected_inputs) == num_images:
        break
    else:
        preds = torch.softmax(model_ft(inputs), 1)
        uniform = torch.Tensor([1 / len(class_names)] * len(class_names))

        if torch.norm(preds - uniform).item() < threshold:
            selected_inputs.append(inputs[0])
            selected_labels.append(labels[0])
            selected_preds.append(preds[0])

print(f"Unsure about these:")
for i in range(num_images):
    plot_prediction(selected_inputs[i].cpu().data, selected_labels[i],
                    selected_preds[i])

# %%

# Build confusion matrix

confusion = torch.zeros(len(class_names), len(class_names))

# FIXME this is terribly inefficient, increase batch size
for i, (inputs, labels) in enumerate(full_val_loader):
    pred = torch.argmax(model_ft(inputs)).item()
    label = labels[0]
    confusion[label][pred] += 1

df_confusion = pd.DataFrame(confusion, columns=class_names,
                            index=class_names).astype(int)
sn.heatmap(df_confusion, annot=True, fmt='d')
plt.xlabel("ground truth")
plt.ylabel("prediction")
plt.show()
