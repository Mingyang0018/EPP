# 介绍

这个存储库包含了论文“A general deep learning model for predicting small molecule products of enzymatic reactions“的代码和数据集。

## 文件说明

这个存储库包含以下文件夹：

    ├── data
    ├── figures
    ├── model
    ├── notebooks_and_code
    ├── requirements.txt
    ├── README_EN.md
    └── README.md

所有代码都包含在"notebooks_and_code"文件夹中。所有生成的文件都在"data", "figures", and "model"文件夹中，分别包含了数据集，图片和训练的模型。所有代码都在UBUNTU系统中运行。

## 使用方式

### 环境安装

首先下载本仓库：

```shell
git clone https://github.com/Mingyang0018/EPP.git
cd EPP
```

建议使用conda创建虚拟环境，然后激活环境。

```shell
conda create -n EPP python=3.10
conda activate EPP
```

然后安装python3, 推荐python 3.10。

安装torch和cuda, 模型训练使用了4个RTX4090, 推荐torch 2.3.1 + cuda 12.1:

conda安装

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

pip安装

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

然后使用 pip 安装依赖：

```shell
pip install -r requirements.txt
```

在终端中输入以下命令，启动Jupyter Notebook:

```shell
jupyter notebook
```

### 代码调用

在Jupyter Notebook中按顺序运行notebooks_and_code中的notebook文件代码，也可以单独运行任何一个notebook文件中的代码。notebook文件说明如下：

- 01_data_preprocessing.ipynb: 获取数据并处理数据
- 02_model_training.ipynb: 在训练数据上训练模型
- 03_model_evaluation.ipynb: 在测试数据上评估模型
- 04_model_prediction.ipynb: 使用训练的模型在新数据上进行预测

## 引用

如果你觉得我们的工作有帮助，请考虑引用下列论文。

```
@article{yang2024EPP,
  title={EPP: A general deep learning model for predicting small molecule products of enzymatic reactions},
  author={Mingxuan Yang, Dong Men and others},
  journal={},
  year={2024}
}
```
