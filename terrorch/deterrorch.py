# beta, error mitigation
import torch
import torch.nn as nn


class Defender():
    @classmethod
    def add_clip(cls, model: nn.Module, **kwargs):

        class ClippedModel(nn.Module):
            def __init__(self, model):
                super(ClippedModel, self).__init__()
                self.model = model

            def forward(self, x):
                for module in self.model.modules():
                    if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU) or isinstance(module, nn.ELU):
                        x = nn.functional.relu(x)
                        x = torch.clamp(x, min=0, max=1)
                    elif isinstance(module, nn.Sigmoid):
                        x = torch.sigmoid(x)
                        x = torch.clamp(x, min=0, max=1)
                    elif isinstance(module, nn.Tanh):
                        x = torch.tanh(x)
                        x = torch.clamp(x, min=-1, max=1)
                    else:
                        x = module(x)
                return x

        return ClippedModel(model)
    
    @classmethod
    def sbp(cls, error_map, **kwargs):
        pass
