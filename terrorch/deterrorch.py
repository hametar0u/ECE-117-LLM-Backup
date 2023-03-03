#beta, error mitigation
import time
import torch
import torch.nn as nn

class Defender():

  def __init__(self, 
      components: list,
      device: torch.device = torch.device('cpu'),
      verbose: bool = False,
      ) -> None:

    self.components = components
    self.device = device
    self.verbose = verbose

    self._argument_validate()
  
  def _argument_validate(self) -> None:
    if self.verbose == True:
      print('Defender initialized.\nProtected components:', self.components)

  def add_protection(self, model: nn.Module):
    
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