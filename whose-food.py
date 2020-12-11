import os

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from PIL import ImageFile
from torchvision import models

import analysis
import data_prep
import network
import util

# %%

# Work around some truncated images: https://stackoverflow.com/a/23575424
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%

# prepare data

data_dir = "data"
all_photos_dir = os.path.join(data_dir, "photos")
split_photos_dir = os.path.join(data_dir, "from")

df = data_prep.load_data(os.path.join(data_dir, "_chat.txt"))
data_prep.show_plots(df)

dataloaders, dataset_sizes, class_names = \
    data_prep.prepare_loaders(split_photos_dir)

# %%

# View some data

# Get a batch of training data
inputs, classes = next(iter(dataloaders["train"]))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

util.imshow(out, title=[class_names[x] for x in classes])
plt.show()

# %%

# load existing model
model = torch.load('results/models/full_model-resnet18_v1.pt')

# %%

# or create new model
base_model = models.resnet18
model = network.build_model(base_model, class_names, True, False)

# %%

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# %%

# train the model

model = network.train_model(model, criterion, optimizer, scheduler,
                            dataloaders, dataset_sizes, num_epochs=1)

# %%

# visualize some parts
util.visualize_model(model, dataloaders, class_names, num_images=6)


# %%

# maybe save the model
torch.save(model, 'full_model-resnet18_feature_extractor_v1.pt')

#%%

# Do some analysis

# Plot some classics
classics = analysis.load_classics(model, dataloaders, class_names)

# Show the confusion
analysis.build_confusion_matrix(model, dataloaders, class_names, plot=True)

# Analyse the activations
analysis.analyse_activations(model, classics, model.layer1, 0)
