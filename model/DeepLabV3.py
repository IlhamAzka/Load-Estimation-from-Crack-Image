from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

def initDeepLabv3(out_channels=1):
    
    # for masks output channels = 1

    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, out_channels)

    # set model in training mode
    model.train()

    return model