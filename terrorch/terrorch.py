import warnings
import torch
import torch.nn as nn

class Injector():
  valid_dtypes = [torch.float, ]
  valid_error_models = ['bit', 'value']
  valid_error_types = ['random', 'stuck_at_fault']

  @classmethod
  def _error_map_generate(cls, injectee_shape: tuple, dtype_bitwidth: int, device: torch.device, p: float) -> torch.Tensor:
    """_summary_

    Args:
        injectee_shape (tuple): _description_
        dtype_bitwidth (int): _description_
        device (torch.device): _description_
        p (float): _description_

    Returns:
        torch.Tensor: _description_
    """
    error_map = (2 * torch.ones((*injectee_shape, dtype_bitwidth), dtype = torch.int, device = device)) ** torch.arange(0, dtype_bitwidth, dtype = torch.int, device = device).expand((*injectee_shape, dtype_bitwidth))
    filter = nn.functional.dropout(torch.ones_like(error_map , dtype = torch.float, device = device), 1 - p)
    error_map  = (filter.int() * error_map).sum(dim = -1).int()
    return error_map

  def __init__(self, 
      p: float = 1e-10, 
      dtype: torch.dtype = torch.float,
      param_names: list = ['weight'],
      device: torch.device = torch.device('cpu'),
      verbose: bool = False,
      error_model = 'bit',
      error_type = 'random',
      ) -> None:
    """_summary_

    Args:
        p (float, optional): _description_. Defaults to 1e-10.
        dtype (torch.dtype, optional): _description_. Defaults to torch.float.
        param_names (list, optional): _description_. Defaults to ['weight'].
        device (torch.device, optional): _description_. Defaults to torch.device('cpu').
        verbose (bool, optional): _description_. Defaults to False.
        error_model (str, optional): _description_. Defaults to 'bit'.
        error_type (str, optional): _description_. Defaults to 'random'.
    """

    self.p = p
    self.dtype = dtype
    self.param_names = param_names
    self.device = device
    self.verbose = verbose
    self.error_model = error_model
    self.error_type = error_type

    self._argument_validate()
    self._dtype_bitwidth = torch.finfo(self.dtype).bits
    self._error_maps = {}
  
  def _argument_validate(self) -> None:
    if self.p <= 0 or self.p >= 1:
      raise ValueError('Invalid probability of error injection.')
    if self.dtype not in Injector.valid_dtypes:
      raise ValueError('Invalid data types. Currently support:', *Injector.valid_dtypes)
    if self.error_model not in Injector.valid_error_models:
      raise ValueError('Unknown error model. Currently support:', *Injector.valid_error_models)
    if self.error_type not in Injector.valid_error_types:
      raise ValueError('Unknown error type. Currently support:', *Injector.valid_error_types)
    if self.verbose == True:
      print('Injector initialized.\nError probability:', self.p)
      print('Data type:', self.dtype)
      print('Error model:', self.error_model)
      print('Error type:', self.error_type)

  
  def _error_map_allocate(self, model: nn.Module) -> None:
    """_summary_

    Args:
        model (nn.Module): _description_
    """
    if self.error_type == 'random':
      self._maxsize = 0
      for param_name, param in model.named_parameters():
        if param_name.split('.')[-1] in self.param_names:
          if param.numel() * self._dtype_bitwidth > self._maxsize:
            self._maxsize = param.numel() * self._dtype_bitwidth
      injectee_shape = (self._maxsize,)
      self._error_maps['universal'] = Injector._error_map_generate(injectee_shape, self._dtype_bitwidth, self.device, self.p)

    elif self.error_type == 'stuck_at_fault':
      for param_name, param in model.named_parameters():
        if param_name.split('.')[-1] in self.param_names:
          injectee_shape = param.shape
          self._error_maps[param_name] = Injector._error_map_generate(injectee_shape, self._dtype_bitwidth, self.device, self.p)
      warnings.warn('Stuck-at-fault error injection is extremely memory-intensive. Use with caution!')

  def inject(self, model: nn.Module) -> None:
    """_summary_

    Args:
        model (nn.Module): _description_
    """
    self._error_map_allocate(model)
    if self.error_type == 'random':
      for param_name, param in model.named_parameters():
        if param_name.split('.')[-1] in self.param_names:
          error_map = self._error_maps['universal']
          error_mask = error_map[torch.randperm(error_map.numel(), device = self.device)][:param.numel()]
          error_mask = error_mask.reshape_as(param)
          param.data = (param.view(torch.int) ^ error_mask).view(torch.float)
    elif self.error_type == 'stuck_at_fault':
      for param_name, param in model.named_parameters():
        if param_name in self._error_maps.keys():
          error_mask = self._error_maps[param_name]
          param.data = (param.view(torch.int) ^ error_mask).view(torch.float)
    if self.verbose == True:
      injected_params = self._error_maps.keys()
      print('The following parameters have been injected:')
      print(injected_params)

  def save_error_map(self, path: str, sparse = False) -> None:
    """_summary_

    Args:
        path (str): _description_
        sparse (bool, optional): _description_. Defaults to False.
    """
    error_maps = self._error_maps.copy()
    for _, v in error_maps.items():
      if self.device != torch.device('cpu'):
        v = v.cpu()
      if sparse == True:
        v = v.to_sparse()
    torch.save(error_maps, path)
    del error_maps
    if self.verbose == True:
      print('Error map saved to:', path)
  
  def load_error_map(self, path: str, sparse = False) -> None:
    """_summary_

    Args:
        path (str): _description_
        sparse (bool, optional): _description_. Defaults to False.
    """
    error_maps = torch.load(path)
    for _, v in error_maps.items():
      if self.device != torch.device('cpu'):
        v = v.to(self.device)
      if sparse == True:
        v = v.to_dense()
    self._error_maps = error_maps.copy()
    del error_maps
    if self.verbose == True:
      print('Error map loaded from:', path)