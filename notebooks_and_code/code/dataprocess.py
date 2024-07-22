# 导入所需的库
# Import necessary libraries
import pandas as pd
import numpy as np
import time
import json
import pubchempy as pcp
from pubchempy import PubChemHTTPError
from rdkit import Chem
from rdkit.Chem import Descriptors
from d2l import torch as d2l
from torch.utils.data import Dataset, DataLoader
import re

# 定义类DataProcess, 包含共用函数
# Define class DataProcess，including shared functions
class DataProcess:
    """
    DataProcess类用于处理化学数据。

    属性:
        molecular_weight_cache (dict): 存储计算过的分子量的字典。
        name_to_smiles_cache (dict): 存储化学分子名称-SMILES字符串的字典。
        smiles_to_name_cache (dict): 存储SMILES字符串-化学分子名称的字典。

    方法:
        __init__(self, path_name_to_smiles_cache):
            类的构造函数，用于初始化属性。

        calculate_molecular_weight(self, smiles):
            计算给定SMILES字符串的分子量。

        sort_smiles_by_mw(self, smiles_arr):
            将SMILES字符串数组按照分子量由大到小排序。

        df_to_smiles(self, df, cols):
            将分子转化为SMILES字符串，并添加至原始dataframe。

        handle_name_to_smiles(self, name):
            处理化学分子名称转化为SMILES字符串时可能出现的异常。

        name_to_smiles(self, name):
            将化学分子名称转化为SMILES字符串。

        add_iupac_ame(self, df):
            在df数组中添加分子的iupac名称。
            
        smiles_to_name(self, smiles):
            将SMILES字符转为化学分子名称。
            
        load_json(path):
            从json文件中读取数据。

        save_json(dict, path):
            将字典保存为json文件。
            
        get_test_data(path, data, cols=['Sequence']):
            获取去重data相同列之后的测试数据。
            
        generate_vocab(data):
            由数据集生成SMILES词汇表
    """
    
    def __init__(self,path_name_to_smiles_cache):
        # 初始化分子量缓存字典
        # Initialize molecular weight cache dictionary
        self.molecular_weight_cache = {}
        # 加载SMILES缓存name_to_smiles_cache_*.json
        # Load SMILES cache name_to_smiles_cache_*.json
        self.name_to_smiles_cache=self.load_json(path=path_name_to_smiles_cache)
        # 初始化smiles_to_name_cache字典
        # Initialize smiles_to_name_cache dictionary
        self.smiles_to_name_cache={}

    def calculate_molecular_weight(self, smiles):
        """
        计算smiles分子的分子量
        Calculate the molecular weight
        """
        # 如果元素为空
        # If the element is empty
        if not smiles:
            return -1
        # 如果元素在字典中
        # If the element is in the dictionary
        elif smiles in self.molecular_weight_cache:
            mw = self.molecular_weight_cache[smiles]
        # 如果元素不在字典中
        # If the element is not in the dictionary
        else:
            mol = Chem.MolFromSmiles(smiles)
            try:
                mw = Descriptors.ExactMolWt(mol)
            except:
                try:
                    # 使用pubchempy的get_compounds函数，根据SMILES字符串搜索PubChem Compound数据库，返回一个Compound对象的列表
                    # Use pubchempy's get_compounds function to search the PubChem Compound database based on the SMILES string and return a list of Compound objects.
                    compounds = pcp.get_compounds(smiles, "smiles")
                    # 如果搜索成功，获取第一个Compound对象的分子量
                    # If search success, get the molecular weight of the first Compound object
                    mw = float(compounds[0].molecular_weight)
                # 如果搜索失败，将分子量设为0
                # If search fails, set the molecular weight to 0
                except:
                    mw = 0
                    print("error:", smiles)
            self.molecular_weight_cache[smiles] = mw
        return mw

    def sort_smiles_by_mw(self, smiles_arr):
        '''
        将smiles_arr数组按照分子量由大到小排序
        Sort the smiles_arr array from large to small molecular weight
        '''
        mol_info = [(smiles, self.calculate_molecular_weight(smiles))
                    for smiles in smiles_arr]
        mol_info.sort(key=lambda x: x[1], reverse=True)
        return np.array([x[0] for x in mol_info])
    
    def df_to_smiles(self, df, cols):
        '''
        将分子转化为SMILES字符串，并添加至原始dataframe
        Convert the molecule into SMILES strings and add it to the original dataframe
        '''
        df.replace(np.nan, "", inplace=True)
        substrate_product_smiles = [
            [self.handle_name_to_smiles(name) for name in row]
            for row in df[cols].values
        ]
        return pd.merge(
                df,
                pd.DataFrame(substrate_product_smiles, columns=cols),
                left_index=True,
                right_index=True,
            )
            # name_to_smiles_cache,

    def handle_name_to_smiles(self, name, retries=10):
        for _ in range(retries):
            try:
                return self.name_to_smiles(name)
            except PubChemHTTPError as e:
                print(f"PubChemHTTPError: {e}, Retrying after 1 seconds...")
                time.sleep(1)
        return None

    def name_to_smiles(self, name, mode=0):
        '''
        将化学分子名称转化为SMILES字符串
        Convert chemical molecule names into SMILES strings
        '''
        smiles = None
        name = str(name).strip()
        if name in self.name_to_smiles_cache:
            return self.name_to_smiles_cache[name]
        elif name:
            results = pcp.get_compounds(name, "name")
            if results:
                smiles = results[0].canonical_smiles
                time.sleep(0.1)
                if mode == 1:
                    smiles = results[0].isomeric_smiles
        self.name_to_smiles_cache[name] = smiles
        return smiles
    
    def add_iupac_ame(self, df):
        '''
        在df数组中添加分子的iupac名称
        Add iupac name of the molecule in dataframe array
        '''
        # 添加分子名称
        # Add molecule name
        for i in range(1, 4):
            df[f"product{i}_pred_name"] = df[f"product{i}_pred"].apply(
                self.smiles_to_name
            )
        return df
    
    def smiles_to_name(self, smiles):
        '''
        将SMILES字符转为化学分子名称
        Convert smiles string into chemical molecule names
        '''
        # 初始化name
        # initialize name
        name = None
        smiles = str(smiles).strip()
        try:
            if smiles in self.smiles_to_name_cache:
                name = self.smiles_to_name_cache[smiles]
            elif (smiles == "<pad>") or (smiles is None) or (smiles == ''):
                name = None
            elif smiles:
                compounds = pcp.get_compounds(smiles, "smiles")
                # 取第一个匹配的化合物的英文名称
                # Get the English name of the first matching compound
                if compounds:
                    name = compounds[0].iupac_name
                else:
                    name = None
            self.smiles_to_name_cache[smiles] = name
            return name
        except Exception as e:
            print(f"Error: {e}")
            return None
        
    @staticmethod
    def load_json(path):
        ''' 
        从json文件中读取内容
        Read content from json file
        '''
        with open(path, "r") as file:
            dict = json.load(file)
        return dict
    
    @staticmethod
    def save_json(dict, path):
        '''
        保存json文件
        Save json file
        '''
        with open(path, "w") as file:
            json.dump(dict, file)
            
    @staticmethod
    def get_test_data(path, data, cols=['Sequence']):
        '''
        获取去重data相同列之后的测试数据
        Obtain test data after deduplicating the same columns of data
        '''
        # 读取数据集
        # Read the dataset
        if path.endswith(".csv"):
            test_data = pd.read_csv(path)
        elif path.endswith(".tsv"):
            test_data = pd.read_csv(path, sep="\t")
        elif path.endswith(".xlsx"):
            test_data = pd.read_excel(path)
        test_data.replace(np.nan, "", inplace=True)

        test_data = test_data[
            ~test_data[cols].apply(
                tuple, axis=1).isin(
                    data[cols].apply(tuple, axis=1)
                )
        ]
        test_data.drop_duplicates(inplace=True)
        test_data.reset_index(inplace=True, drop=True)
        return test_data
    
    @staticmethod
    def generate_vocab(data):
        '''
        # 由数据集生成SMILES词汇表
        # Generate SMILES vocabulary from dataset
        '''
        product_smiles_tokens = d2l.tokenize(
            pd.concat(
                [
                    data['reaction1_product1_y'],
                    data['reaction1_product2_y'],
                    data['reaction1_product3_y'],
                    data['reaction1_product4_y'],
                    data['reaction1_product5_y']
                ]
            ),
            token="word",
        )
        product_smiles_vocab = d2l.Vocab(
            product_smiles_tokens,
            min_freq=1,
            reserved_tokens=["<pad>", "<bos>", "<eos>", "<sep>"],
        )
        # print(len(product_smiles_vocab), product_smiles_vocab.token_to_idx)
        return product_smiles_vocab
    
    
# 定义MyDatasetPredict类
# Define MyDatasetPredict class
class MyDatasetPredict(Dataset):
    def __init__(
        self,
        data,
        product_smiles_vocab,
        prot_tokenizer,
        chemBERTa_tokenizer,
        max_seq_length,
        max_mol_length,
        dataProcess,
    ):
        self.data = data
        self.product_smiles_vocab = product_smiles_vocab
        self.prot_tokenizer = prot_tokenizer
        self.chemBERTa_tokenizer = chemBERTa_tokenizer
        self.max_seq_length = max_seq_length
        self.max_mol_length = max_mol_length
        self.dataProcess = dataProcess

    def __getitem__(self, index):
        row = self.data.iloc[index]
        sequence = row["Sequence"]
        substrate1 = row.get('reaction1_substrate1_y', '')
        substrate2 = row.get('reaction1_substrate2_y', '')
        substrate3 = row.get('reaction1_substrate3_y', '')
        substrate4 = row.get('reaction1_substrate4_y', '')
        substrate5 = row.get('reaction1_substrate5_y', '')
        substrate6 = row.get('reaction1_substrate6_y', '')
        substrate7 = row.get('reaction1_substrate7_y', '')
        substrate8 = row.get('reaction1_substrate8_y', '')
        substrate9 = row.get('reaction1_substrate9_y', '')
        substrate10 = row.get('reaction1_substrate10_y', '')

        # 对酶序列进行编码，得到prot_model的输入向量
        # Encode the enzyme sequence to get the input vector of prot_model
        # 在序列中插入空格并替换非标准氨基酸
        # Insert spaces and replace non-standard amino acids in the sequence
        sequences = [
            " ".join(re.sub(r"[UZOB]", "X", sequence)) for sequence in [sequence]
        ]
        # 对序列进行分词和编码
        # Segment and encode the sequence
        sequence_tokens = self.prot_tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
        )
        # 获取编码向量
        # Get encoding vector
        sequence_input_ids = sequence_tokens["input_ids"].clone().detach()
        sequence_attention_mask = sequence_tokens["attention_mask"].clone(
        ).detach()

        # 对底物进行编码，得到chemBERTa的输入向量
        # Encode the substrate and obtain the input vector of chemBERTa
        # 对底物SMILES字符串进行分词和编码
        # Segment and encode the substrate SMILES string
        substrates = [
            self.dataProcess.sort_smiles_by_mw(
                np.array(
                    [
                        substrate1,
                        substrate2,
                        substrate3,
                        substrate4,
                        substrate5,
                        substrate6,
                        substrate7,
                        substrate8,
                        substrate9,
                        substrate10,
                    ]
                )
            )
            for substrate1, substrate2, substrate3, substrate4, substrate5, substrate6, substrate7, substrate8, substrate9, substrate10 in [
                (
                    substrate1,
                    substrate2,
                    substrate3,
                    substrate4,
                    substrate5,
                    substrate6,
                    substrate7,
                    substrate8,
                    substrate9,
                    substrate10,
                )
            ]
        ]
        substrates = [
            f"{substrate1}<sep>{substrate2}<sep>{substrate3}"
            for substrate1, substrate2, substrate3, _, _, _, _, _, _, _ in substrates
        ]

        substrate_tokens = self.chemBERTa_tokenizer.batch_encode_plus(
            substrates,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_mol_length,
            truncation=True,
        )
        # 获取编码向量
        # Get encoding vector
        substrate_input_ids = substrate_tokens["input_ids"].clone().detach()
        substrate_attention_mask = substrate_tokens["attention_mask"].clone(
        ).detach()

        return (
            sequence_input_ids,
            sequence_attention_mask,
            substrate_input_ids,
            substrate_attention_mask,
        )

    def __len__(self):
        return len(self.data)