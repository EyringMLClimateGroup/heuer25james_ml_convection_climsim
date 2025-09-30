# Beyond the Training Data: Confidence-Guided Mixing of Parameterizations in a Hybrid AI-Climate Model

This repository contains the code for the developement of data-driven convection parameterizations based on the NARVAL data set for ICON-A

The corresponding paper is available as a preprint on arXiv
> ...

Corresponding DOI:

[![DOI](https://zenodo.org/badge/1052597208.svg)](https://zenodo.org/badge/latestdoi/1052597208)

We adapted the model and parts of the training pipeline from the [leap-climsim-kaggle-5th repository](https://github.com/YusefAN/leap-climsim-kaggle-5th), which presents a solution that achieved 5th place in the 2024 [ClimSim Kaggle competition](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview).

After cloning, initialize submodules with:
```
git submodule update --init --recursive
```

## Repository content
### [training](training): contains the training script for the model
- `training/BiLSTM.py` contains the BiLSTM model code
- `training/train_export_model.ipynb` contains the code for training the model
- `training/hpo.py` contains the hyperparameter optimization code
### [preprocessing](preprocessing): contains python scripts and notebooks to create pre-processed data
- `preprocessing/snapshot_download.py` for downloading the [ClimSim high-res](https://huggingface.co/datasets/LEAP/ClimSim_high-res) dataset
- Contains [rte+rrtmgp](preprocessing/rte-rrtmgp) git submodule which contains scripts on how to subtract radiative tendencies from the ClimSim columns in `examples/all-sky/`. The file `compute_tend.sh` is a starting point to compute these tendencies.
- The preprocessing files are mostly adapted versions from the [ClimSim repository](https://github.com/leap-stc/ClimSim/tree/main)
- `preprocessing/adding_input_feature.ipynb` computes the files with expanded input features
- `preprocessing/create_dataset_example_v5expandcnv.ipynb` computes the training/validation/testing files
- `preprocessing/normalize_data_v5expandcnv.ipynb` for normalization of the data
- `preprocessing/normalization` contains normalization constants
### [online_evaluation](online_evaluation): contains various notebooks to create the figures used in the manuscript

- Contains various notebooks for analyzing the coupled online runs
- Many of these online evaluations use the processed files created by the [ESMValTool](https://github.com/ESMValGroup/ESMValTool)

## Data and Coupling
All training data used in the study is publicly available at [huggingface](https://huggingface.co/datasets/LEAP/ClimSim_high-res). More details about the data can be found [here](https://arxiv.org/abs/2306.08754).

For the online-coupling of the ML models we used the Ftorch library (https://github.com/Cambridge-ICCS/FTorch)

## Dependencies
All used dependencies can be found in the `env.yml` file and can be installed with mamba/conda using the command: `conda env create --name <your_env_name> --file=env.yml`. 

The main Dependencies included are:
- PyTorch
- PyTorch Lightning
- Scikit-learn
- Xarray
- Netcdf4
- Dask
- Ray
- Numba
- Other packages like Numpy, Matplotlib, Pandas, and Tqdm
