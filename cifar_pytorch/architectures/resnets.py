import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.utils import load_state_dict_from_url


class MyWideResnet(ResNet):
    def __init__(self, block, layers, withfc, **kwargs):
        super(MyWideResnet, self).__init__(block, layers, **kwargs)
        self.withfc = withfc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.withfc:
            x = self.fc(x)

        return x


def _resnet(block, layers, withfc, **kwargs):
    model = MyWideResnet(block, layers, withfc, **kwargs)
    return model


def custom_wrn(layers=None, withfc=False, width_per_group=64*2, **kwargs):
    if layers is None:
        layers = [3, 4, 6, 3]
    kwargs['width_per_group'] = width_per_group
    return _resnet(Bottleneck, layers, withfc, **kwargs)


def wide_resnet50_2(pretrained, progress=True, withfc=False):
    model = custom_wrn(layers=[3, 4, 6, 3], withfc=withfc)
    if pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
                                              progress=progress)
        model.load_state_dict(state_dict)
    return models.wide_resnet50_2(pretrained=pretrained)