# Sparse vs Contiguity

This repository represents the replication code for the study "Sparse vs Contiguity".

## Table of Contents

- [Getting Started](#getting-started)
- [Information regarding the parameters](#information-regarding-the-parameters)

## Getting Started
1. Download the official datset ImageNet from the original webseite. 
2. The install the required packages using the following commands:
### Download and install Miniforge (an equivalent of Miniconda)
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O ~/miniforge.sh
bash ~/miniforge.sh -b -p ~/miniforge3
```
### Activate base env and run init for the future
```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate
conda init
```
### Create conda environment
```bash
conda env create --file requirements.yaml
```
### Delete installer
```bash
rm ~/miniforge.sh
```
Note: The file requirements.yaml contains all the required dependencies


3. Run the script `gen_datast.py` such that it creates the specific dataset of the preprocessed images. 
4. After that you can create specific configurations in the ``\config``. As an example you can check the 2 files in ``\config``.
5. You can run the attacks as follows, but you also need to be in the ``\code`` directory!!!:
```bash
python runner.py test_clip_sparse
```

## Information regarding the parameters:

### Sparse attack (One-pixel in the code)
  - Mandatory type of the attack!!!! ``type_attack: one_pixel2`` 
  - pretrained_model: the string which represents the model (check the ``model_dict.py`` to see the respective string for each model)
  - dataset: the string which represents the dataset (available: ``cifar10``, ``cifar100``, ``imagenet``)
  - generate_captions: bool value which is True if captions where generated, False otherwise
  - type_target: the string which represents the type of the target -> ``second``/``random``
  - save_plot: the boolean value which is True if we want to save the plots, False otherwise
  - pixels: the number of perturbed pixels
  - popsize: the population size
  - maxiter: the number of max iterations
  - crossover: the crossover rate
  - mutation: the mutation rate
  - fitness_function: the string which represents the fitness function (check the ``fit.py`` to see the respective string for each model)
  - strategy:  the string which represents the strategy used in the DE (check the ``differential_evolution.py`` to see the respective string for each model)
  - polish: the boolean value which is True if the LBFGS was used, False otherwise
  - max_iter_lbfgs: the maximum number of iterations in the LBGFS
  - targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
  - stats: the boolean value which is True if we want to save the stats, False otherwise
  - code: the string which represents the code of the experiment, such that it can be easier identifiable
  - log_file: the name of the log file
  - log_type: type of the log(``info``, ``debug``, ``critical``)

### Random Sparse attack (random_one in the code)
  - Mandatory type of the attack!!!! ``type_attack: random_one``  
  - pretrained_model: the string which represents the model (check the ``model_dict.py`` to see the respective string for each model)
  - dataset: the string which represents the dataset (available: ``cifar10``, ``cifar100``, ``imagenet``)
  - generate_captions: bool value which is True if captions where generated, False otherwise
  - type_target: the string which represents the type of the target -> ``second``/``random``
  - save_plot: the boolean value which is True if we want to save the plots, False otherwise
  - pixels: the number of perturbed pixels
  - popsize: the population size
  - maxiter: the number of max iterations
  - fitness_function: the string which represents the fitness function (check the ``fit.py`` to see the respective string for each model)
  - targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
  - stats: the boolean value which is True if we want to save the stats, False otherwise
  - code: the string which represents the code of the experiment, such that it can be easier identifiable
  - log_file: the name of the log file
  - log_type: type of the log(``info``, ``debug``, ``critical``)


### Contiguous attack (patch in the code)
  - Mandatory type of the attack!!!! ``type_attack: one_anti_diag``, ``type_attack: one_diag`` , ``type_attack: one_row``,  ``type_attack: one_column``, ``type_attack: one_patch``
  - If the Patch attack is used, just modify the w and h values (the number of pixels can be arbitrary since the system computes the correct number)
  - If other attack is used, just modify the pixels (the numbers for w and h can be arbitrary since the system computes the correct number)
  - pretrained_model: the string which represents the model (check the ``model_dict.py`` to see the respective string for each model)
  - dataset: the string which represents the dataset (available: ``cifar10``, ``cifar100``, ``imagenet``)
  - generate_captions: bool value which is True if captions where generated, False otherwise
  - type_target: the string which represents the type of the target -> ``second``/``random``
  - save_plot: the boolean value which is True if we want to save the plots, False otherwise
  - pixels: the number of perturbed pixels
  - popsize: the population size
  - maxiter: the number of max iterations
  - crossover: the crossover rate
  - mutation: the mutation rate
  - fitness_function: the string which represents the fitness function (check the ``fit.py`` to see the respective string for each model)
  - strategy:  the string which represents the strategy used in the DE (check the ``differential_evolution.py`` to see the respective string for each model)
  - w: the integer which represents the width
  - h: the integer which represents the height
  - targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
  - stats: the boolean value which is True if we want to save the stats, False otherwise
  - code: the string which represents the code of the experiment, such that it can be easier identifiable
  - log_file: the name of the log file
  - log_type: type of the log(``info``, ``debug``, ``critical``)

### Random Contiguous attack (random_patch in the code)
  - Mandatory type of the attack!!!! ``type_attack: random_anti_diag``, ``type_attack: random_diag`` , ``type_attack: random_row``,  ``type_attack: random_column``, ``type_attack: random_patch``
  - If the Random Patch attack is used, just modify the w and h values (the number of pixels can be arbitrary since the system computes the correct number)
  - If other attack is used, just modify the pixels (the numbers for w and h can be arbitrary since the system computes the correct number)
  - pretrained_model: the string which represents the model (check the ``model_dict.py`` to see the respective string for each model)
  - dataset: the string which represents the dataset (available: ``cifar10``, ``cifar100``, ``imagenet``)
  - generate_captions: bool value which is True if captions where generated, False otherwise
  - type_target: the string which represents the type of the target -> ``second``/``random``
  - save_plot: the boolean value which is True if we want to save the plots, False otherwise
  - pixels: the number of perturbed pixels
  - popsize: the population size
  - maxiter: the number of max iterations
  - w: the integer which represents the width
  - h: the integer which represents the height
  - fitness_function: the string which represents the fitness function (check the ``fit.py`` to see the respective string for each model)
  - targeted: the bool value which indicates targeted attacks (True) and non-targeted attacks (False)
  - stats: the boolean value which is True if we want to save the stats, False otherwise
  - code: the string which represents the code of the experiment, such that it can be easier identifiable
  - log_file: the name of the log file
  - log_type: type of the log(``info``, ``debug``, ``critical``)