# 导入所需的库
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw

import torch
from torch import nn
from torch.nn import functional as F
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

from fire import Fire
from openpyxl import Workbook
from io import BytesIO
import openpyxl
import seaborn as sns
# 设置调色板
# Set palette
sns.set_palette("muted")
# 设置环境变量为false
# Set the environment variable to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入模型文件
# Import model files
from model import Net

# 导入数据处理文件
# Import dataprocess files
from dataprocess import DataProcess, MyDatasetPredict

def predict(model, data, dataProcess, number_label, device, batch_size):
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
        num_workers=batch_size,
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
    batch_size=16,
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
        test_data = dataProcess.df_to_smiles(test_data, cols=[f'reaction1_substrate{i}' for i in range(1,1+NUMBER_REACTION)])
    # 在测试数据上预测
    # Predict on test data
    product_pred = predict(model=model, data=test_data, dataProcess=dataProcess, number_label=number_label, device=device, batch_size=batch_size)
    
    # 拼接测试数据和预测结果
    # Concatenate test data and prediction results
    df_result = pd.concat([test_data, product_pred], axis=1)

    return df_result

def smiles_to_molecule(smiles):
    '''
    从SMILES创建分子对象并添加氢和坐标
    Create a molecule object from SMILES and add hydrogens and coordinates
    '''
    if (smiles == "<pad>") or (smiles is None):
        return None
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # mol = Chem.AddHs(mol)
            # 计算二维坐标
            AllChem.Compute2DCoords(mol)
            # 计算三维坐标
            # AllChem.EmbedMolecule(mol)
        return mol
    
# 保存添加分子结构图的模型预测结果为EXCEL文件
# Save the model prediction results of added molecular structure diagrams as excel files
def smiles_to_excel(df, cols, save_path):
    '''
    在dataframe中添加分子结构图，并保存为EXCEL文件
    Add the molecular structure diagram to dataframe and save it as an excel file
    '''
    # 创建一个EXCEL
    # Create an EXCEL
    wb = Workbook()
    ws = wb.active
    
    # 添加标题行
    # Add title row
    img_cols = ['img_'+x for x in cols]
    titles = df.columns.values.tolist() + img_cols
    ws.append(titles)
    
    # 添加df
    # add dataframe
    for index, row in df.iterrows():
        ws.append(row.tolist())
                    
    # 生成SMILES的图像并添加到EXCEL
    # Generate SMILES structure diagrams and add them to EXCEL
    df01 = df[cols]
    arr_smiles_activate = np.empty([len(cols)])
    # 遍历SMILES列的行
    # Traverse the rows of SMILES column
    for i in range(df01.shape[0]):
        # 如果当前行与上一行的SMILES不一致
        # If the SMILES of the current line is inconsistent with the previous line
        if not (df01.iloc[i,:].values == arr_smiles_activate).all():
            arr_smiles_activate = df01.iloc[i,:].values
            for j in range(df01.shape[1]):
                smiles = df01.iloc[i,j]
                # 判断smiles是否有效
                # Determine whether smiles are valid
                if isinstance(smiles, str) and (smiles.strip()) and (smiles != '<pad>'):
                    # 生成化学分子的二维结构图
                    # Generate two-dimensional structure diagram of a chemical molecule
                    mol = Chem.MolFromSmiles(smiles)
                    img = Draw.MolToImage(mol)
                    
                    # 将图像转化为字节流
                    # Convert image to byte stream
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # 在EXCEL中插入图像
                    # Insert image into EXCEL
                    img_cell = ws.cell(row=i+2, column=len(df.columns)+j+1)
                    img = openpyxl.drawing.image.Image(BytesIO(img_byte_arr))
                    ws.add_image(img, img_cell.coordinate)
                
    # 保存excel文件
    # Save excel file
    wb.save(save_path)
    # 打印生成了预测结果至save_path
    # Print generated prediction to save path
    print("generated prediction to: ", save_path)

# 定义主函数mymain
def mymain(path=None):
    # 读取数据集
    data01 = pd.read_csv(path_CA_SMILES, sep="\t")
    data01.replace(np.nan, "", inplace=True)

    # 生成词汇表
    product_smiles_vocab = dataProcess.generate_vocab(data=data01)

    # 获取path，如果为空，则使用默认路径
    # Get path, if empty, use default path
    if path is None:
        path_input = os.path.join(parent_dir,'data/data_input_template.xlsx')
    else:
        path_input = path
    # 获取path_input文件名及后缀
    # Get the filename and suffix of path_input
    file_name, file_extension = os.path.splitext(path_input)
    save_path = file_name + '_pred' + file_extension
    test_input = dataProcess.get_test_data(path=path_input, data=data01, cols=['Sequence'])

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

    smiles_to_excel(df=df_pred01, 
                    cols=['reaction1_substrate1_y', 'reaction1_substrate2_y', 'reaction1_substrate3_y', 'product1_pred', 'product2_pred', 'product3_pred',],
                    save_path=save_path
                    )

# 定义具有Catalytic Activity的原始数据文件，DATE后添加01，来区分源于path_source_enzyme的训练集、验证集和测试集文件
# Define the original data file with Catalytic Activity, add '01' after DATE to distinguish the training set, validation set and test set files originating from path_source_enzyme
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATE = "20240611"
DATE += '01'
path_source_CA = os.path.join(parent_dir, f"data/data_reviewed_CA_{DATE}.tsv")
path_CA_SMILES = path_source_CA.replace(f'{DATE}', f'{DATE}_SMILES')

# 创建一个DataProcess对象
path_name_to_smiles_cache = os.path.join(parent_dir,'data/name_to_smiles_cache_20240611.json')
dataProcess = DataProcess(path_name_to_smiles_cache = path_name_to_smiles_cache)
# 定义提取的同一个催化反应的底物/产物的最大数量
# Define the maximum number of substrates/products extracted from the same catalytic reaction
NUMBER_REACTION = 10

# 定义常量
NUMBER_LABEL = 3
BATCH_SIZE = 8*2  # 批次大小
MAX_LEN_MOL = 256  # 底物分子序列最大长度
MAX_LEN_SEQ = 1573  # 氨基酸序列最大长度，序列长度的98%分位数

# 检查是否有可用的GPU，如果有，将设备设置为GPU，否则设置为CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义蛋白语言模型和化学分子语言模型的模型和分词器，并将它们移动到设备上
prot_bert_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
prot_bert_model = AutoModel.from_pretrained("Rostlab/prot_bert")

chemBERTa_MTR_tokenizer = AutoTokenizer.from_pretrained(
    "DeepChem/ChemBERTa-77M-MTR")
chemBERTa_MTR_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

# 执行主函数mymain
if __name__ == '__main__':
    Fire(mymain)