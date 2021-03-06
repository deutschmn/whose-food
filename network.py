import copy
import time

import torch
import torch.nn as nn
import torch.utils.data


def train_model(model, criterion, optimizer, scheduler, dataloaders,
                dataset_sizes, num_epochs=5):
    """
    Train a model
    :param model: model
    :param criterion: loss function
    :param optimizer: optimizer to use
    :param scheduler: LR scheduler
    :param dataloaders: data loaders
    :param dataset_sizes: sizes of the data sets
    :param num_epochs: number of epochs to train
    :return: trained model
    """
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


def build_model(base_model, class_names, pretrained=True,
                freeze_gradients=False):
    """
    Build a model
    :param base_model: basis, should be ResNet
    :param class_names: dict of class names
    :param pretrained: whether the model should be pretrained
    :param freeze_gradients: whether to freeze gradients in training or not
    :return: model
    """
    model = base_model(pretrained=pretrained)
    if freeze_gradients:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    return model
