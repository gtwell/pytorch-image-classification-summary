# 创建待训练的模型
import torchvision
from torch import nn
import pretrainedmodels


def create_model(model_key,
                 pretrained,
                 num_of_classes,
                 device):
    """Create CNN model

    Args:
        model_key (str): Name of the model to be created.
        pretrained: If True, only train the laster layer.
        num_of_classes (int): Number of categories of outputs.

    Returns:
        model_conv: A defined model to be train

    Raises:
        ValueError: Error while asking an unrecognized model.
    """
    if model_key == 'resnet18':
        model_conv = torchvision.models.resnet18(pretrained=True)
    elif model_key == 'resnet34':
        model_conv = torchvision.models.resnet34(pretrained=True)
    elif model_key == 'resnet50':
        model_conv = torchvision.models.resnet50(pretrained=True)
    elif model_key == 'resnet101':
        model_conv = torchvision.models.resnet101(pretrained=True)
    elif model_key == 'inception_v3':
        model_conv = torchvision.models.inception_v3(pretrained=True)
    elif model_key == 'inceptionresnetv2':
        model_conv = pretrainedmodels.inceptionresnetv2(num_classes=1000,
                                                        pretrained='imagenet')
    elif model_key == 'inceptionv4':
        model_conv = pretrainedmodels.inceptionv4(num_classes=1000,
                                                  pretrained='imagenet')
    elif model_key == 'nasnetalarge':
        model_conv = pretrainedmodels.nasnetalarge(num_classes=1000,
                                              pretrained='imagenet')
    else:
        raise ValueError("Unrecognized name of model {}".format(model_key))

    if pretrained:
        # Lock parameters for transfer learning
        for param in model_conv.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    if model_key == 'nasnetalarge' or model_key == 'inceptionresnetv2' or model_key == 'inceptionv4':
        dim_feats = model_conv.last_linear.in_features # 2048
        model_conv.last_linear = nn.Linear(dim_feats, num_of_classes)
    else:
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, num_of_classes)

    # Initialize newly added module parameters
    if model_key == 'nasnetalarge' or model_key == 'inceptionresnetv2' or model_key == 'inceptionv4':
        nn.init.xavier_uniform_(model_conv.last_linear.weight)
        nn.init.constant_(model_conv.last_linear.bias, 0)
    else:
        nn.init.xavier_uniform_(model_conv.fc.weight)
        nn.init.constant_(model_conv.fc.bias, 0)

    model_conv = model_conv.to(device)

    return model_conv
