# Terrorch
- Terrorch: A User-friendly, Efficient, and Flexible Hardware Error Injection Framework

## Highlights
- User-friendly:
    - Completely PyTorch-based. No other dependency required ever. (It is recommended use packages such as `timeit` and `tqdm` for auxiliary purposes and running the examples / demos.)
    - No need to compile, build or even use custom environments.
- Efficient:
    - Uses `torch.view()` for bit flip operations, which significantly (about 100X) accelerates the error injection effort compared with other error injection tools. 
- Flexible:
    - Works for any PyTorch model as long as they are implemented as `torch.nn.Module` with named parameters.
    - Implements two mitigation methods for emulation: activation filtering and selective bit protection in `deterrorch.py` and allows user customization.
    - Allows user to define their customized error model such as random value error beyond the provided bit flip error. 

## Usage Examples / Demos
- See notebooks in `./examples`
    - (Quick Start) Error injection to Vision Transformers: `./examples/ViT.ipynb`
    - Evaluating and Enhancing Robustness of Deep Recommendation Systems Against Silent Hardware Errors: `./examples/DRS/`