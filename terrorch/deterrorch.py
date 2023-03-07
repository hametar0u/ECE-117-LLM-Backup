# beta, error mitigation
import torch
import torch.nn as nn

class Defender():
    """This class defines the error mitigation schemes (not exhaustive). Add your custom error mitigations as classmethod here.
    """    

    @classmethod
    def _output_limitation(cls, model: nn.Module, **kwargs) -> nn.Module:
        raise NotImplementedError('Activation limitation is not implemented in Defender. Please directly use Injector._activation_limitation()!')
    
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
