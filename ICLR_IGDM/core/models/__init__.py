import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .wideresnet import wideresnet

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish

from core.data import DATASETS


MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101', 
          'wrn-28-10', 'wrn-32-10', 'wrn-34-10', 'wrn-34-20', 
          'preact-resnet18-swish', 'preact-resnet34-swish', 
          'wrn-28-10-swish', 'wrn-34-10-swish', 'wrn-34-20-swish', 'wrn-70-16-swish']


def create_model(name, normalize = False):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    backbone = wideresnet(name, num_classes=10)
    model = torch.nn.Sequential(backbone)
    
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    return model
