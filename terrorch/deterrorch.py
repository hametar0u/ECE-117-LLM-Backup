# beta, error mitigation
import torch
import torch.nn as nn

class Defender():

    @classmethod
    def add_clip(cls, model: nn.Module, **kwargs) -> nn.Module:
        """This method applies a clamp function to limit the the output of activation functions within a range as a kind of error mitigation.

        Args:
            model (nn.Module): The target model.

        Returns:
            nn.Module:: The model after adding clipping.
        """
        class ClippedModel(nn.Module):
            def __init__(self, model):
                super(ClippedModel, self).__init__()
                self.model = model

            def forward(self, x):
                for module in self.model.modules():
                    if isinstance(module, nn.ReLU):
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
    def sbp(cls, error_maps: dict, **kwargs) -> None:
        """Sanitize the error map according to selected bit positions for error mitigation.

        Args:
            error_map (dict): The error map for selective bit protection
        """        
        error_maps_ = error_maps.copy()
        sbp_mask = ~(torch.tensor([1 << i for i in kwargs['protected_bits']]).sum())
        for param_name, param in error_maps_.items():
            error_maps_[param_name] = param & sbp_mask
        return error_maps_
