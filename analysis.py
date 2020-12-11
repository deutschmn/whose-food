import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
import torch.utils.data
import torchvision

import util


def load_classics(model, dataloaders, class_names):
    # TODO make more efficient (instead of loading one at a time)
    full_val_loader = dataloaders['full_val']

    threshold = 0.99
    num_images = 6

    all_inputs = []

    for class_name in class_names:
        selected_inputs = []
        selected_preds = []

        class_idx = class_names.index(class_name)
        for i, (inputs, labels) in enumerate(full_val_loader):
            if labels[0] == class_idx:
                if len(selected_inputs) == num_images:
                    break
                else:
                    preds = torch.softmax(model(inputs), 1)

                    if preds[0][class_idx] >= threshold:
                        selected_inputs.append(inputs[0])
                        selected_preds.append(preds[0])

        print(f"Classic {class_name}:")
        for i in range(len(selected_inputs)):
            util.plot_prediction(selected_inputs[i].cpu().data, class_idx,
                                 selected_preds[i], class_names)
        all_inputs.extend(selected_inputs)
    return torch.stack(all_inputs)


def load_unsures(model, dataloaders, class_names):
    full_val_loader = dataloaders['full_val']
    threshold = 0.3
    num_images = 6

    selected_inputs = []
    selected_labels = []
    selected_preds = []

    for i, (inputs, labels) in enumerate(full_val_loader):
        if len(selected_inputs) == num_images:
            break
        else:
            preds = torch.softmax(model(inputs), 1)
            uniform = torch.Tensor([1 / len(class_names)] * len(class_names))

            if torch.norm(preds - uniform).item() < threshold:
                selected_inputs.append(inputs[0])
                selected_labels.append(labels[0])
                selected_preds.append(preds[0])

    print(f"Unsure about these:")
    for i in range(num_images):
        util.plot_prediction(selected_inputs[i].cpu().data, selected_labels[i],
                             selected_preds[i], class_names)


def build_confusion_matrix(model, dataloaders, class_names, plot=True):
    # TODO this is terribly inefficient, increase batch size
    full_val_loader = dataloaders['full_val']
    confusion = torch.zeros(len(class_names), len(class_names))

    for i, (inputs, labels) in enumerate(full_val_loader):
        pred = torch.argmax(model(inputs)).item()
        label = labels[0]
        confusion[label][pred] += 1

    df_confusion = pd.DataFrame(confusion, columns=class_names,
                                index=class_names).astype(int)

    if plot:
        sn.heatmap(df_confusion, annot=True, fmt='d')
        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.show()

    return df_confusion


def analyse_activations(model, inputs, layer, image_idx):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    outputs = []

    def track_outputs(_, __, output):
        print("track_outputs was called")
        outputs.append(output)

    hook = layer[0].register_forward_hook(track_outputs)

    with torch.no_grad():
        predictions = model(inputs)

    print([output.shape for output in outputs])

    image = outputs[0][image_idx]

    activation = image.norm(2, dim=0)

    util.show_activation(activation)
    plt.show()

    out = torchvision.utils.make_grid(inputs)
    util.imshow(out)
    plt.show()

    hook.remove()


