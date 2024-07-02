## Introduction

This repository contains the code and dataset for the paper "A general deep learning model for predicting small molecule products of enzymatic reactions".

## File Structure

This repository contains the following folders:

    ├── data
    ├── figures
    ├── model
    ├── notebooks_and_code
    ├── requirements.txt
    ├── README_EN.md
    └── README.md

All code is included in the "notebooks_and_code" folder. All generated files are stored in the "data", "figures", and "model" folders, which contain the dataset, images, and trained models, respectively. All code is run on a UBUNTU system.

## Usage

### Environment Setup

First, clone this repository:

```shell
git clone https://github.com/Mingyang0018/EPP.git
cd EPP
```

Install Python 3.10 or later.

It is recommended to use conda to create a virtual environment and activate it.

```shell
conda create -n EPP python=3.10
conda activate EPP
```

Install torch and CUDA, the model training used 4 RTX4090 cards, recommend torch 2.3.1 + CUDA 12.1:

conda installation:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

pip installation:

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then, install the required dependencies using pip:

```shell
pip install -r requirements.txt
```

Start Jupyter Notebook in the terminal:

```shell
jupyter notebook
```

### Code Execution

Run the notebook files in the "notebooks_and_code" folder in order, or run any individual notebook file's code separately. The notebook files are as follows:

- 01_data_preprocessing.ipynb: Obtain and preprocess data
- 02_model_training.ipynb: Train the model on training set
- 03_model_evaluation.ipynb: Evaluate the model on testing set
- 04_model_prediction.ipynb: Use the trained model to make predictions on new data

## Citation

If you find our work helpful, please consider citing the following paper:

```
@article{yang2024EPP,
  title={EPP: A general deep learning model for predicting small molecule products of enzymatic reactions},
  author={Mingxuan Yang, Dong Men and others},
  journal={},
  year={2024}
}
```
