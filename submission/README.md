This submission directory is organized as follows:
```
- code
- pytei
- results
```
`code` contains the final version of the notebook we used to run experiments. `pytei` contains the PyTEI source code, along with the error model modifications we made. `results` contain our raw results.

## Set Up Instructions
 - Install the dependencies in `requirements.yml`:
 ```bash
    conda env create --name PyTEI --file=requirements.yml
    conda activate PyTEI
 ```
 - Run `fork-of-injectornotebook.ipynb`