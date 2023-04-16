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
 - Install the dependencies in `requirements.yml`:
 ```bash
    conda env create --name terrorch --file=requirements.yml
 ```
 - Download the datasets:
   - MovieLens-1M: https://grouplens.org/datasets/movielens/1m/
   - MovieLens-20M: https://grouplens.org/datasets/movielens/20m/
   - Criteo DAC: https://ailab.criteo.com/ressources/ 

- Examples: see notebooks in `./examples`
    - (Quick Start) Error injection to Vision Transformers: `./examples/ViT.ipynb`
    - Evaluating and Enhancing Robustness of Deep Recommendation Systems Against Silent Hardware Errors: `./examples/DRS/`
    
## To-do:
- [X] Allow users to determine the device of generating and injecting errors.
- [X] Automatic error map size allocation by iterating through model.
- [X] Support for save and load of error maps with option to save as sparse tensor.
- [X] Implement for stuck-at-fault error (with customized error map).
- [X] Implement error mitigation methods of activation limiting and selective bit protection (SBP).
- [ ] Allow users to specify individual parameters for error mitigation.
- [ ] Support for random value error.
- [ ] Support for 64-bit double precision.
- [ ] Support error injection during training.
- [ ] Support for quantized data types.

## Contact
- Dongning Ma, Ph.D. Student, VU-DETAIL, email: dma2@villanova.edu 
- Richard Wang, Ph.D. Student, VU-DETAIL, email: rwang8@villanova.edu 
- Sizhe Zhang, Ph.D. Student, VU-DETAIL, email: szhang6@villanova.edu 
- Prof. Xun Jiao, Advisor, VU-DETAIL, email: xun.jiao@villanova.edu 
