import torch
import torch.nn as nn

class Injector():
  valid_dtypes = [torch.float, ]

  def __init__(self, *args, **kwargs) -> None:
    self.p = kwargs.get('p', 1e-10)
    self.dtype = kwargs.get('dtype', torch.float)
    self.param_names = kwargs.get('param_names')
    self.device = kwargs.get('device')
  
  def _argument_validate(self) -> None:
    if self.p <= 0 or self.p >= 1:
      raise ValueError('Invalid probability of error injection.')
    if self.dtype not in Injector.valid_dtypes:
      raise ValueError('Invalid data types.')
    if self.device is None:
      raise ValueError('Please specify device for error injection.')
  
  def _error_map_generate(self) -> None:
    self._error_map = torch.ones((self.maxsize, torch.finfo(self.dtype).bits), device = self.device)
    self._error_map = (2 * torch.ones(32, dtype = torch.int, device = self.device)) ** torch.arange(0, 32, dtype = torch.int, device = self.device).expand_as(self._error_map)
    filter = nn.functional.dropout(torch.ones_like(self._error_map, dtype = torch.float, device = self.device), 1 - self.p)
    self._error_map = filter.int() * self._error_map 
    self._error_map = self._error_map.sum(dim = -1).int()

  def inject(self, model: nn.Module) -> None:
    self._errormap_size_detect(model)
    self._error_map_generate()
    for param_name, param in model.named_parameters():
      if param_name.split('.')[-1] in self.param_names:
        error_mask = self._error_map[torch.randperm(self._error_map.numel(), device = self.device)][:param.numel()]
        error_mask = error_mask.reshape_as(param)
        param.data = (param.view(torch.int) ^ error_mask).view(torch.float)
  
  def _errormap_size_detect(self, model: nn.Module) -> None:
    self.maxsize = 0
    for param_name, param in model.named_parameters():
      if param_name.split('.')[-1] in self.param_names:
        if param.numel() * torch.finfo(self.dtype).bits > self.maxsize:
          self.maxsize = param.numel() * torch.finfo(self.dtype).bits

  def save_errormap(self, path, sparse = False) -> None:
    maptensor = self._error_map.clone()  
    if self.device != torch.device('cpu'):
      maptensor = maptensor.cpu()
    if sparse == True:
      maptensor = maptensor.to_sparse() 
    torch.save(maptensor, path)
  
  def load_errormap(self, path, sparse = False) -> None:
    maptensor = torch.load(path)
    if self.device != torch.device('cpu'):
      maptensor = maptensor.to(self.device)
    if sparse == True:
      maptensor = maptensor.to_dense()
    self._error_map = maptensor.clone()