import time
import torch
import torch.nn as nn
from terrorch.deterrorch import Defender

class Injector():
    valid_dtypes = [torch.float, ]
    valid_error_models = ['bit', 'value']
    valid_mitigations = ['None', 'SBP', 'clip']

    @classmethod
    def _error_map_generate(cls, injectee_shape: tuple, dtype_bitwidth: int, device: torch.device, p: float) -> torch.Tensor:
        """Injecting bit errors into the tensor based on the given parameters.

        Args:
            injectee_shape (tuple): The shape of the tensor that is the target of the error injection
            dtype_bitwidth (int): The bits that the each element in the provided data type occupies.
            device (torch.device): The device on which the error injection is carried out.
            p (float): The probability of the error.

        Returns:
            torch.Tensor: The tensor with error injected.
        """
        error_map = (2 * torch.ones((*injectee_shape, dtype_bitwidth), dtype=torch.int, device=device)) ** torch.arange(
            0, dtype_bitwidth, dtype=torch.int, device=device).flip(dims=(-1, )).expand((*injectee_shape, dtype_bitwidth))
        filter = (p * nn.functional.dropout(torch.ones_like(error_map,
                  dtype=torch.float, device=device), 1 - p)).int()
        error_count = filter.sum(dim=-1)
        error_map = (filter * error_map).sum(dim=-1).int()
        return error_map, error_count

    @classmethod
    def _error_map_generate_v(cls, injectee_shape: tuple, dtype_bitwidth: int, device: torch.device, p: float) -> torch.Tensor:
        """Injecting randome value errors into the tensor based on the given parameters.

        Args:
            injectee_shape (tuple): The shape of the tensor that is the target of the error injection
            dtype_bitwidth (int): The bits that the each element in the provided data type occupies.
            device (torch.device): The device on which the error injection is carried out.
            p (float): The probability of the error.

        Returns:
            torch.Tensor: The tensor with error injected.
        """
        raise NotImplementedError(
            'Random value error model is not implemented yet.')

    def __init__(self,
                 p: float = 1e-10,
                 dtype: torch.dtype = torch.float,
                 param_names: list = ['weight'],
                 device: torch.device = torch.device('cpu'),
                 verbose: bool = False,
                 error_model='bit',
                 mitigation = 'None',
                 ) -> None:
        """The initialization of the Injector class.

        Args:
            p (float, optional): The probability of the error. Defaults to 1e-10.
            dtype (torch.dtype, optional): The data type of the target model. Defaults to torch.float.
            param_names (list(str), optional): The parameters that wished to be injected with error. Defaults to ['weight'].
            device (torch.device, optional): The device on which the error injection is carried out. Defaults to torch.device('cpu').
            verbose (bool, optional): Setting True to print information about error injection. Defaults to False.
            error_model (str, optional): The error model. Defaults to 'bit'.
        """
        self.p = p
        self.dtype = dtype
        self.param_names = param_names
        self.device = device
        self.verbose = verbose
        self.error_model = error_model
        self.mitigation = mitigation

        self._argument_validate()
        self._dtype_bitwidth = torch.finfo(self.dtype).bits
        self._error_maps = {}
        self._error_count = {}

    def _argument_validate(self) -> None:
        if self.p <= 0 or self.p >= 1:
            raise ValueError('Invalid probability of error injection.')
        if self.dtype not in Injector.valid_dtypes:
            raise ValueError('Invalid data types.')
        if self.error_model not in Injector.valid_error_models:
            raise ValueError('Unknown error model.')
        if self.mitigation not in Injector.valid_mitigations:
            raise ValueError('Unknown mitigation method.')
        if self.verbose == True:
            print('Injector initialized.\nError probability:', self.p)
            print('Data type:', self.dtype)
            print('Error model:', self.error_model)

    def _error_map_allocate(self, model: nn.Module) -> None:
        """Iterative through model parameters and allocate the error maps for injection.

        Args:
            model (nn.Module): The target model for error injection.
        """
        if type(model) == dict:
            for param_name, param in model.items():
                for each_param in self.param_names:
                    if each_param in param_name:
                        injectee_shape = param.shape
                        self._error_maps[param_name], self._error_count[param_name] = Injector._error_map_generate(
                            injectee_shape, self._dtype_bitwidth, self.device, self.p)
                        break
        else:
            for param_name, param in model.named_parameters():
                for each_param in self.param_names:
                    if each_param in param_name:
                        injectee_shape = param.shape
                        self._error_maps[param_name], self._error_count[param_name] = Injector._error_map_generate(
                            injectee_shape, self._dtype_bitwidth, self.device, self.p)
                        break

    def inject(self, model: nn.Module) -> None:
        """Injecting the errors into the model

        Args:
            model (nn.Module): The target model for error injection.
        """
        start_time = time.time()
        self._error_map_allocate(model)
        error_count_number = 0
        param_count_number = 0

        if type(model) == dict:
            for param_name, param in model.items():
                if param_name in self._error_maps.keys():
                    error_mask = self._error_maps[param_name]
                    error_count_number += self._error_count[param_name].sum()
                    param_count_number += self._error_maps[param_name].numel()
                    param.data = (param.view(torch.int) ^
                                  error_mask).view(torch.float)
        else:
            for param_name, param in model.named_parameters():
                if param_name in self._error_maps.keys():
                    error_mask = self._error_maps[param_name]
                    error_count_number += self._error_count[param_name].sum()
                    param_count_number += self._error_maps[param_name].numel()
                    param.data = (param.view(torch.int) ^
                                  error_mask).view(torch.float)

        if self.verbose == True:
            injected_params = self._error_maps.keys()
            print('The following parameters have been injected:')
            print(injected_params)
            print('Total number of errors injected:', error_count_number)
            print('Total number of parameters:', param_count_number)
            print('Time spent on error injection (second):',
                  time.time() - start_time)

    def save_error_map(self, path: str, sparse=False) -> None:
        """Save error map as a file.

        Args:
            path (str): The path for saving the error map file.
            sparse (bool, optional): Setting True for saving under sparse format of tensor. Defaults to False.
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

    def load_error_map(self, path: str, sparse=False) -> None:
        """Load error map from a file.

        Args:
            path (str): The path of the error map file to load.
            sparse (bool, optional): Setting True for loading error maps saved under sparse format of tensor. Defaults to False.
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
    
    def _activation_limitation(self, **kwargs) -> None:
        """This method adds forward hooks to each named parameter to check if there are invalid activation values.
        """  
        def clamp_output(module, input, output):
            output.clamp_(min = kwargs['min'], max = kwargs['max'])
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(clamp_output)

    def config_mitigation(self) -> None:
        """This method is to handle different types of error mitigation schemes.
        """                
        if self.mitigation != None:
            if self.mitigation == 'SBP':
                self._error_maps = Defender.sbp(self._error_maps, protected_bits = [31, 30, 29, 28]) #sign bit and top 3 exponent bits
            if self.mitigation == 'clip':
                self._activation_limitation(min = -1000, max = 1000) #very loose limitation actually
