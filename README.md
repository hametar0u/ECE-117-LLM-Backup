# Terrorch
- Terrorch: A User-friendly, Efficient, and Flexible Hardware Error Injection Framework

## Highlights
- User-friendly: Terrorch prioritizes user-friendliness by relying on the PyTorch framework only, eliminating the need for any external dependencies. Unlike other similar frameworks, it reduces the hassle of dealing with installing additional packages or libraries, and configuring dockers and environment, allowing users to quickly and conveniently utilize the tool. While it is recommended to use auxiliary packages such as `timeit` and `tqdm` for supplementary purposes and when running examples or demos, they are not mandatory prerequisites.
- Efficient: To achieve better efficiency in error injection, the tool leverages the Pytorch built-in`torch.view()` function for bit flip operations. This approach yields a substantial acceleration in the error injection process, achieving a speed enhancement of approximately 100 times compared to other frameworks using `struct` or `bitstring` packages. By adopting this technique, users can significantly reduce the time spent on their experiments or simulations, leading to expedited results.
- Flexible: The tool offers impressive flexibility as it supports any PyTorch model that inherits the `torch.nn.Module` class with named parameters using PyTorch canonical tensors, regardless of the complexity or architecture of the user's model. In addition, existing frameworks hardly incorporate interface for implementing mitigation methods. Terrorch by default implements two mitigation methods for emulation, namely the activation filtering and the selective bit protection. Those mitigation methods are implemented within the `deterrorch.py` module and can be further customized by users. The tool provides users with the ability to define their own error models, error maps and mitigation methods. This feature extends beyond the default bit flip errors and enables other error models, such as random value errors and stuck-at faults.

## Usage Examples / Demos
 - Install the dependencies in `requirements.yml`:
 ```bash
    conda env create --name terrorch --file=requirements.yml
 ```

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
