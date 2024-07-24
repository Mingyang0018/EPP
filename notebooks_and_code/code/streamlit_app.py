# 这是一个streamlit应用，用于快速使用EPP模型，输入蛋白序列和底物，点击预测，显示预测的产物及其概率。

# 导入所需的库
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os

import pubchempy as pcp
# from pubchempy import PubChemHTTPError
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw

from sympy import sequence
import torch
from torch import nn
from torch.nn import functional as F
# import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

import seaborn as sns
# 设置调色板
# Set palette
sns.set_palette("muted")
# 设置环境变量为false
# Set the environment variable to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import base64

# 导入模型文件
# Import model files
from model import Net
# 导入数据处理文件
# Import dataprocess files
from dataprocess import DataProcess, MyDatasetPredict

def predict(model, data, dataProcess, number_label, device, batch_size=1):
    '''
    模型在新数据上预测
    Model predicts on new data
    '''
    model.eval()
    product_preds = torch.empty(0, number_label).to(device)
    product_probs = torch.empty(0, number_label).to(device)

    # 创建MyDatasetPredict实例
    # Create MyDatasetPredict instance
    dataset = MyDatasetPredict(
        data=data,
        product_smiles_vocab=model.product_smiles_vocab,
        prot_tokenizer=model.prot_tokenizer,
        chemBERTa_tokenizer=model.chemBERTa_tokenizer,
        max_seq_length=model.max_seq_length,
        max_mol_length=model.max_mol_length,
        dataProcess=dataProcess,
    )
    # 创建DataLoader实例
    # Create DataLoader instance
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=batch_size,
        pin_memory=True,
    )

    # 遍历批次
    # Iterate through batch
    for (
        sequence_input_ids,
        sequence_attention_mask,
        substrate_input_ids,
        substrate_attention_mask,
    ) in dataloader:
        # 如果序列为空，跳过该批次
        # If the sequence is empty, skip the batch
        if sequence_input_ids is None:
            continue
        with torch.no_grad():
            (
                sequence_input_ids,
                sequence_attention_mask,
                substrate_input_ids,
                substrate_attention_mask,
            ) = (
                sequence_input_ids.to(device),
                sequence_attention_mask.to(device),
                substrate_input_ids.to(device),
                substrate_attention_mask.to(device),
            )

            # 调用Net类的前向传播方法，得到预测的催化产物向量
            # Call the forward propagation method of Net class to obtain the predicted catalytic product vector
            predicted_vectors1, predicted_vectors2, predicted_vectors3 = model(
                sequence_input_ids,
                sequence_attention_mask,
                substrate_input_ids,
                substrate_attention_mask,
            )

            # 对预测向量进行归一化，得到概率分布
            # Normalize the prediction vector to obtain the probability distribution
            predicted_probs1 = F.softmax(predicted_vectors1, dim=1)
            predicted_probs2 = F.softmax(predicted_vectors2, dim=1)
            predicted_probs3 = F.softmax(predicted_vectors3, dim=1)

            # 对概率分布进行取最大值，得到类别索引和概率值
            # Take the maximum value of the probability distribution and obtain the category index and probability value
            predicted_values1, predicted_ids1 = torch.max(
                predicted_probs1, dim=1)

            predicted_values2, predicted_ids2 = torch.max(
                predicted_probs2, dim=1)

            predicted_values3, predicted_ids3 = torch.max(
                predicted_probs3, dim=1)

        product_pred = torch.cat(
            (
                predicted_ids1.unsqueeze(1),
                predicted_ids2.unsqueeze(1),
                predicted_ids3.unsqueeze(1),
            ),
            dim=1,
        )
        product_prob = torch.cat(
            (
                predicted_values1.unsqueeze(1),
                predicted_values2.unsqueeze(1),
                predicted_values3.unsqueeze(1),
            ),
            dim=1,
        )

        product_preds = torch.cat((product_preds, product_pred), dim=0)
        product_probs = torch.cat((product_probs, product_prob), dim=0)
        # 手动清理GPU内存
        # Manually clear GPU memory
        torch.cuda.empty_cache()
        
    df_preds = pd.DataFrame(
        product_preds.cpu().detach().numpy(),
        columns=["product1_pred", "product2_pred", "product3_pred"],
    )
    df_preds['product1_pred'] = df_preds['product1_pred'].map(lambda x: model.product_smiles_vocab.idx_to_token[int(x)])
    df_preds['product2_pred'] = df_preds['product2_pred'].map(lambda x: model.product_smiles_vocab.idx_to_token[int(x)])
    df_preds['product3_pred'] = df_preds['product3_pred'].map(lambda x: model.product_smiles_vocab.idx_to_token[int(x)])
    df_probs = pd.DataFrame(
        product_probs.cpu().detach().numpy(),
        columns=["product1_prob", "product2_prob", "product3_prob"],
    )
    df_result = pd.concat([df_preds, df_probs], axis=1)
    return df_result

def model_predict(
    net,
    test_data,
    dataProcess,
    device,
    folder_name="model",
    model_name="model01.pth",
    number_label=3,
    batch_size=1,
    is_smiles=False,
):
    '''
    加载模型，并在新数据上预测
    Load the model and predict on new data
    '''
    # 加载模型
    # Load model
    MODEL_FOLDER = folder_name
    path_savemodel = f"{MODEL_FOLDER}/{model_name}"
    print(path_savemodel)
    model = net.to(device)
    state_dict = torch.load(path_savemodel, map_location=device)['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict, strict=False)

    # 将化学分子转化为SMILES字符
    # Convert chemical molecules into SMILES
    test_data.replace(np.nan, "", inplace=True)
    if is_smiles == False:
        test_data = dataProcess.df_to_smiles(test_data, cols=[f'reaction1_substrate{i}' for i in range(1,4)])
    # 在测试数据上预测
    # Predict on test data
    product_pred = predict(model=model, data=test_data, dataProcess=dataProcess, number_label=number_label, device=device, batch_size=batch_size)
    
    # 拼接测试数据和预测结果
    # Concatenate test data and prediction results
    df_result = pd.concat([test_data, product_pred], axis=1)

    return df_result

# 定义具有Catalytic Activity的原始数据文件，DATE后添加01，来区分源于path_source_enzyme的训练集、验证集和测试集文件
# Define the original data file with Catalytic Activity, add '01' after DATE to distinguish the training set, validation set and test set files originating from path_source_enzyme
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATE = "2024061101"
path_source_CA = os.path.join(parent_dir, f"data/data_reviewed_CA_{DATE}.tsv")
path_CA_SMILES = path_source_CA.replace(f'{DATE}', f'{DATE}_SMILES')

# 创建一个DataProcess对象
path_name_to_smiles_cache = os.path.join(parent_dir,'data/name_to_smiles_cache_20240611.json')
dataProcess = DataProcess(path_name_to_smiles_cache = path_name_to_smiles_cache)

# # 定义提取的同一个催化反应的底物/产物的最大数量
# # Define the maximum number of substrates/products extracted from the same catalytic reaction
# NUMBER_REACTION = 10

# 定义常量
NUMBER_LABEL = 3
BATCH_SIZE = 1  # 批次大小
MAX_LEN_MOL = 256  # 底物分子序列最大长度
MAX_LEN_SEQ = 1573  # 氨基酸序列最大长度，序列长度的98%分位数

# 检查是否有可用的GPU，如果有，将设备设置为GPU，否则设置为CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集
data01 = pd.read_csv(path_CA_SMILES, sep="\t")
data01.replace(np.nan, "", inplace=True)

# 生成词汇表
product_smiles_vocab = dataProcess.generate_vocab(data=data01)

# 显示主界面
# Display the main interface
st.set_page_config(layout="centered")

# 设置背景图片
# Set background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
set_png_as_page_bg(os.path.join(parent_dir, 'figures/back20240719.png'))

def set_font_color():
    font_style = '''
    <style>
    .stApp header h2, .stApp div, .stApp p {
        color: white;
    }
    </style>
    '''
    st.markdown(font_style, unsafe_allow_html=True)
    
set_font_color()

# 设置header，字体颜色为白色
# Set header, font color is white
st.header(':white[EPP酶产物预测]', help='根据蛋白序列和底物，预测酶催化产物', divider='rainbow')

# 显示蛋白序列输入文本框，允许多行显示
# Display the protein sequence input text box, allowing multiple lines of display
sequence = st.text_area("请输入蛋白序列:", key='sequence')

# 显示底物输入文本框
# Display substrates input text box
st.text("请输入底物(仅支持小分子):")
sub_col1, sub_col2, sub_col3 = st.columns(3)
# 输入底物1
# Input substrate1
substrate1 = sub_col1.text_input('底物1:', key='substrate1', label_visibility='collapsed')

# 如果输入substrate1，则显示substrate1的分子二维图
if substrate1:
    smiles1 = dataProcess.name_to_smiles(name=substrate1)
    sub_col1.text(smiles1)
    compound1 = pcp.get_compounds(smiles1, 'smiles')
    # 显示分子式
    # Display molecular formula
    sub_col1.text(compound1[0].molecular_formula)
    # 显示smiles1的二维图
    # Display the two-dimensional diagram of smiles1
    mol1 = Chem.MolFromSmiles(smiles1)
    img1 = Draw.MolToImage(mol1)
    sub_col1.image(img1, use_column_width=True
            )
substrate2 = sub_col2.text_input('底物2', key='substrate2', label_visibility='collapsed')
if substrate2:
    smiles2 = dataProcess.name_to_smiles(name=substrate2)
    sub_col2.text(smiles2)
    compound2 = pcp.get_compounds(smiles2, 'smiles')
    # 显示分子式
    # Display molecular formula
    sub_col2.text(compound2[0].molecular_formula)
    # 显示smiles1的二维图
    # Display the two-dimensional diagram of smiles1
    mol2 = Chem.MolFromSmiles(smiles2)
    img2 = Draw.MolToImage(mol2)
    sub_col2.image(img2, use_column_width=True
            )
substrate3 = sub_col3.text_input('底物3', key='substrate3', label_visibility='collapsed')
if substrate3:
    smiles3 = dataProcess.name_to_smiles(name=substrate3)
    sub_col3.text(smiles3)
    compound3 = pcp.get_compounds(smiles3, 'smiles')
    # 显示分子式
    # Display molecular formula
    sub_col3.text(compound3[0].molecular_formula)
    # 显示smiles1的二维图
    # Display the two-dimensional diagram of smiles1
    mol3 = Chem.MolFromSmiles(smiles3)
    img3 = Draw.MolToImage(mol3)
    sub_col3.image(img3, use_column_width=True
            )

# 添加预测按钮
# Add prediction button
if st.button('预测', type='primary', key='predict'):
    # 整理输入数据为dataframe数组，列名为 ['Sequence', 'substrate1', 'substrate2', 'substrate3']
    # Organize input data into dataframe array, column names ['Sequence', 'substrate1', 'substrate2', 'substrate3']
    test_input = pd.DataFrame([[sequence, substrate1, substrate2, substrate3]], columns=['Sequence', 'reaction1_substrate1', 'reaction1_substrate2', 'reaction1_substrate3'])

    # 定义蛋白语言模型和化学分子语言模型的模型和分词器，并将它们移动到设备上
    prot_bert_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    prot_bert_model = AutoModel.from_pretrained("Rostlab/prot_bert")

    chemBERTa_MTR_tokenizer = AutoTokenizer.from_pretrained(
        "DeepChem/ChemBERTa-77M-MTR")
    chemBERTa_MTR_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    net01 = Net(
        prot_bert_model,
        prot_bert_tokenizer,
        chemBERTa_MTR_model,
        chemBERTa_MTR_tokenizer,
        product_smiles_vocab,
        MAX_LEN_SEQ,
        MAX_LEN_MOL,
        BATCH_SIZE
    )
    df_pred01 = model_predict(
        net=net01,
        test_data=test_input,
        dataProcess=dataProcess,
        folder_name=os.path.join(parent_dir,'model'),
        model_name="model01_151k_80.pth",
        device=device,
        number_label=NUMBER_LABEL,
        batch_size=BATCH_SIZE,
        is_smiles=False
    )

    df_pred01 = dataProcess.add_iupac_ame(df=df_pred01)

    # 设置4列布局
    # Set 4-column layout
    st.subheader(':white[预测产物:]', divider='rainbow', help='分数>0.9时，预测较准确')
    col0, col1, col2, col3 = st.columns([1,2,2,2])
    # 设置左侧标题
    # Set left title
    col0.text('分数: ')
    col0.text('名称: ')
    col0.text('SMILES: ')
    col0.text('分子式: ')
    col0.text('结构图: ')
    # 显示预测产物1
    # Display the predicted product1
    if (df_pred01['product1_pred'].iloc[0] is not None) and (df_pred01['product1_pred'].iloc[0] != '<pad>'):
        col1.text(df_pred01['product1_prob'].iloc[0])
        col1.text(df_pred01['product1_pred_name'].iloc[0])
        col1.text(df_pred01['product1_pred'].iloc[0])
        compound1 = pcp.get_compounds(df_pred01['product1_pred'].iloc[0], 'smiles')
        # 显示分子式
        # Display molecular formula
        col1.text(compound1[0].molecular_formula)
        mol1 = Chem.MolFromSmiles(df_pred01['product1_pred'].iloc[0])
        img1 = Draw.MolToImage(mol1)
        col1.image(img1, use_column_width=True)
    
    # 显示预测产物2
    # Display the predicted product2
    if (df_pred01['product2_pred'].iloc[0] is not None) and (df_pred01['product2_pred'].iloc[0] != '<pad>'):
        col2.text(df_pred01['product2_prob'].iloc[0])
        col2.text(df_pred01['product2_pred_name'].iloc[0])
        col2.text(df_pred01['product2_pred'].iloc[0])
        compound2 = pcp.get_compounds(df_pred01['product2_pred'].iloc[0], 'smiles')
        # 显示分子式
        # Display molecular formula
        col2.text(compound2[0].molecular_formula)
        mol2 = Chem.MolFromSmiles(df_pred01['product2_pred'].iloc[0])
        img2 = Draw.MolToImage(mol2)
        col2.image(img2, use_column_width=True)
        
    # 显示预测产物3
    # Display the predicted product3
    if (df_pred01['product3_pred'].iloc[0] is not None) and (df_pred01['product3_pred'].iloc[0] != '<pad>'):
        col3.text(df_pred01['product3_prob'].iloc[0])
        col3.text(df_pred01['product3_pred_name'].iloc[0])
        col3.text(df_pred01['product3_pred'].iloc[0])
        compound3 = pcp.get_compounds(df_pred01['product3_pred'].iloc[0], 'smiles')
        # 显示分子式
        # Display molecular formula
        col3.text(compound3[0].molecular_formula)
        mol3 = Chem.MolFromSmiles(df_pred01['product3_pred'].iloc[0])
        img3 = Draw.MolToImage(mol3)
        col3.image(img3, use_column_width=True)

    # 显示气球动画
    # Display balloon animation
    st.balloons()