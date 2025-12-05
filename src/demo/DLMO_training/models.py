import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_kaiyan(nn.Module):
#     The denoising network used in Kaiyan's work.

    def __init__(self, dim1=260, dim2=311):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding="same")  # here
        self.conv2 = nn.Conv2d(64, 64, 5, padding="same")
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(int(dim1 * dim2 * 64), 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        # x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x

    def get_layer_output(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        return x


class Net_7conv2_dropout(nn.Module):
# CNN with 7 convolutional layers, dropout, and no pooling.

    def __init__(self, dim1=260, dim2=311, filter_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, filter_size, padding="same")  # here
        self.conv2 = nn.Conv2d(64, 64, filter_size, padding="same")
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(int(dim1 * dim2 * 64), 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        # x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_layer_output(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        return x


class Net_7conv2_filter9_dropout(nn.Module):
# Similar to Net_7conv2_dropout, but with 9x9 filters instead of 7x7.

    def __init__(self, dim1=260, dim2=311):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 9, padding="same")  # here
        self.conv2 = nn.Conv2d(64, 64, 9, padding="same")
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(int(dim1 * dim2 * 64), 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.dropout(x)

        # x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_layer_output(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        return x

def load_radimagenet_weight(model, checkpoint):
    # Load weights from a RadImageNet checkpoint into the given model.
    #
    # Args:
    # model: The model to load weights into
    # checkpoint: The RadImageNet checkpoint dictionary
    #
    # Returns:
    # The model with loaded weights and a new fully connected layer

    new_state_dict = {k.replace('backbone.0', 'conv1'): v for k, v in checkpoint.items()}
    new_state_dict = {k.replace('backbone.1', 'bn1'): v for k, v in new_state_dict.items()}
    new_state_dict = {k.replace('backbone.4', 'layer1'): v for k, v in new_state_dict.items()}
    new_state_dict = {k.replace('backbone.5', 'layer2'): v for k, v in new_state_dict.items()}
    new_state_dict = {k.replace('backbone.6', 'layer3'): v for k, v in new_state_dict.items()}
    new_state_dict = {k.replace('backbone.7', 'layer4'): v for k, v in new_state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)  # strict=False to ignore fc mismatch
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

class Net_4conv2(nn.Module):
#    CNN with 4 convolutional layers, dropout, and no pooling.

    def __init__(self, dim1=260, dim2=311):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding="same")  # here
        self.conv2 = nn.Conv2d(64, 64, 5, padding="same")
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(int(dim1 * dim2 * 64), 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))

        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))

        # x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_layer_output(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x))
        return x