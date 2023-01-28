import torch
import torch.nn as nn

class Injector():
  def __init__(self, *args, **kwargs) -> None:
    self.p = kwargs.get('p', 1e-10)
    self.dtype = kwargs.get('dtype', torch.float)
    self.param_names = kwargs.get('param_names')
    self.device = kwargs.get('device')
    self._error_map_generate()
  
  @classmethod
  def _argument_validate(self) -> None:
    pass
  
  def _error_map_generate(self) -> None:
    self._error_map = torch.ones((self.size, torch.finfo(self.dtype).bits), device = self.device)
    self._error_map = (2 * torch.ones(32, dtype = torch.int, device = self.device)) ** torch.arange(0, 32, dtype = torch.int, device = self.device).expand_as(self._error_map)
    filter = nn.functional.dropout(torch.ones_like(self._error_map, dtype = torch.float, device = self.device), 1 - self.p)
    self._error_map = filter.int() * self._error_map 
    self._error_map = self._error_map.sum(dim = -1).int()

  def inject(self, model: nn.Module) -> None:
    self._errormap_size_detect(model)
    for param_name, param in model.named_parameters():
      if param_name.split('.')[-1] in self.param_names:
        error_mask = self._error_map[torch.randperm(self._error_map.numel())][:param.numel()]
        error_mask = error_mask.reshape_as(param) #check
        param.data = (param.view(torch.int) ^ error_mask).view(torch.float)
  
  def _errormap_size_detect(self, model: nn.Module) -> None:
    self.size = 0
    for param_name, param in model.named_parameters():
      if param_name.split('.')[-1] in self.param_names:
        if param.numel() * torch.finfo(self.dtype).bits > self.size:
          self.size = param.numel() * torch.finfo(self.dtype).bits
  
  def inject_sparse(self, model: nn.Module) -> None:
    return NotImplementedError('Sparse error map is not implemented yet.')
