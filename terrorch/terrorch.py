class Injector():
  def __init__(self, *args, **kwargs) -> None:
    self.p = kwargs.get('p', 1e-10)
    self.dtype = kwargs.get('dtype', torch.float)
    self.size = kwargs.get('max_size')
    self.param_names = kwargs.get('param_names')
    self._error_map_generate()
  
  def _error_map_generate(self) -> None:
    self._error_map = torch.ones((self.size, torch.finfo(self.dtype).bits))
    self._error_map = (2 * torch.ones(32, dtype = torch.int)) ** torch.arange(0, 32, dtype = torch.int).expand_as(self._error_map)
    filter = nn.functional.dropout(torch.ones_like(self._error_map, dtype = torch.float), 1 - self.p)
    self._error_map = filter.int() * self._error_map 
    self._error_map = self._error_map.sum(dim = -1).int()

  def inject(self, model: nn.Module) -> None:
    for param_name, param in model.named_parameters():
      if param_name.split('.')[-1] in self.param_names:
        if param.numel() > self._error_map.numel():
          raise ValueError('Your predefined error map is too small!')
        error_mask = self._error_map[torch.randperm(self._error_map.numel())][:param.numel()]
        error_mask = error_mask.reshape_as(param) #check
        param.data = (param.view(torch.int) ^ error_mask).view(torch.float)
  
  def inject_iterate(self, model: nn.Module) -> None:
    return NotImplementedError('Iterative error injection is not implemented yet.')
  
  def inject_sparse(self, model: nn.Module) -> None:
    return NotImplementedError('Sparse error map is not implemented yet.')
