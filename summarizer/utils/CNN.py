import torch
import torch.nn as nn
from torchvision import transforms, models


class FeatureExtractor(nn.Module):
    """
    Pre-trained CNN acting as a feature extractor
    Args:
        model_name: string, googlenet (default) or any compatible model from
                    torchvision.models
        layer_limit: int, describing the index of the layer to stop at for
                     feature extraction (e.g. pooling layer after a conv layer)

    """

    def __init__(self, model_name='googlenet', layer_limit=-3):
        super(FeatureExtractor, self).__init__()

        self.model_name = model_name
        self.layer_limit = layer_limit

        # rescale and normalize transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        try:
            model_definition = getattr(models, self.model_name)
            model = model_definition(pretrained=True)
        except Exception as e:
            raise Exception(f"Could not load model with name {model_name}: {e}")

        model.float()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            model.cuda()
        model.eval()

        module_list = list(model.children())

        assert self.layer_limit < len(module_list), "Error: layer_limit refers to \
        an out-of-bounds layer"

        self.conv5 = nn.Sequential(*module_list[:self.layer_limit])
        self.pool5 = module_list[self.layer_limit]

    def forward(self, x):
        x = self.transform(x)
        x = x.unsqueeze(0)  # reshape the single image s.t. it has a batch dim of 1
        if self.cuda:
            x = x.cuda()
        out_conv5 = self.conv5(x)
        out_pool5 = self.pool5(out_conv5)
        out_pool5 = out_pool5.view(out_pool5.size(0), -1)
        return out_pool5
