import torch
import torchvision
import torchvision.models as models


def wide_resnet50_2(pretrained):
    return models.wide_resnet50_2(pretrained=pretrained)