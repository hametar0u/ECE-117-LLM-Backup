import torch
import torch.nn as nn

class Injector():
  valid_dtypes = [torch.float, ]
  valid_error_models = ['bit', 'value']

  @classmethod
  def _error_map_generate(injectee_shape, dtype_bitwidth, device, p) -> torch.Tensor:
    error_map = (2 * torch.ones((injectee_shape, dtype_bitwidth), dtype = torch.int, device = device)) ** torch.arange(0, dtype_bitwidth, dtype = torch.int, device = device).expand((injectee_shape, dtype_bitwidth))
    filter = nn.functional.dropout(torch.ones_like(error_map , dtype = torch.float, device = device), 1 - p)
    error_map  = (filter.int() * error_map).sum(dim = -1).int()
    return error_map

  def __init__(self, 
      p: float = 1e-10, 
      dtype: torch.dtype = torch.float,
      param_names: list = ['weight'],
      device: torch.device = torch.device('cpu'),
      verbose: bool = False,
      error_model = 'bit'

      ) -> None:

    self.p = p
    self.dtype = dtype
    self.param_names = param_names
    self.device = device
    self.verbose = verbose
    self.error_model = error_model

    self._argument_validate()
    self._dtype_bitwidth = torch.finfo(self.dtype).bits
    self._error_maps = {}
  
  def _argument_validate(self) -> None:
    if self.p <= 0 or self.p >= 1:
      raise ValueError('Invalid probability of error injection.')
    if self.dtype not in Injector.valid_dtypes:
      raise ValueError('Invalid data types.')
    if self.error_model not in Injector.valid_error_models:
      raise ValueError('Unknown error model.')
  
  def _error_map_allocate(self, model: nn.Module) -> None:
    if self.error_model == 'random':
      self._maxsize = 0
      for param_name, param in model.named_parameters():
        if param_name.split('.')[-1] in self.param_names:
          if param.numel() * self._dtype_bitwidth > self.maxsize:
            self.maxsize = param.numel() * self._dtype_bitwidth

      injectee_shape = self.maxsize
      self._error_maps['universal'] = Injector._error_map_generate(injectee_shape, self._dtype_bitwidth, self.device, self.p)

    elif self.error_model == 'stuck_at_fault':
      for param_name, param in model.named_parameters():
        if param_name.split('.')[-1] in self.param_names:
          injectee_shape = torch.zeros_like(param)
          self._error_maps[param_name] = Injector._error_map_generate(injectee_shape, self._dtype_bitwidth, self.device, self.p)
      raise Warning('Stuck-at-fault error injection is extremely memory-intensive. Use with caution!')

  def inject(self, model: nn.Module) -> None:
    self._error_map_allocate(model)
    if self.error_model == 'random':
      for param_name, param in model.named_parameters():
        if param_name.split('.')[-1] in self.param_names:
          error_mask = self._error_map['universal'][torch.randperm(self._error_map.numel(), device = self.device)][:param.numel()]
          error_mask = error_mask.reshape_as(param)
          param.data = (param.view(torch.int) ^ error_mask).view(torch.float)
    elif self.error_model == 'stuck_at_fault':
      for param_name, param in model.named_parameters():
        if param_name in self.param_names:
          error_mask = self._error_map[param_name]
          param.data = (param.view(torch.int) ^ error_mask).view(torch.float)

  def save_error_map(self, path, sparse = False) -> None:
    error_maps = self._error_maps.copy()
    if self.device != torch.device('cpu'):
      error_maps = [error_map.cpu() for error_map in error_maps]
    if sparse == True:
      error_maps = [error_map.to_sparse()  for error_map in error_maps]
    torch.save(error_maps, path)
  
  def load_error_map(self, path, sparse = False) -> None:
    error_maps = torch.load(path)
    if self.device != torch.device('cpu'):
      error_maps = [error_map.to(self.device) for error_map in error_maps]
    if sparse == True:
      error_maps = [error_map.to_dense() for error_map in error_maps]
    self._error_maps = error_maps.copy()