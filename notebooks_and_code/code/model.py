# 定义DNN模型，同时预测三个product
import torch
import warnings

warnings.filterwarnings("ignore")


# 定义模型Net类
# Define model Net class
class Net(torch.nn.Module):
    # 初始化方法，传入蛋白语言模型和化学分子模型的模型和分词器，定义DNN网络的encoder和decoder
    # Initialization method, pass in the model and word segmenter of the protein language model and chemical molecule model, and define the encoder and decoder of the DNN network
    def __init__(
        self,
        prot_model,
        prot_tokenizer,
        chemBERTa_model,
        chemBERTa_tokenizer,
        product_smiles_vocab,
        max_seq_length,
        max_mol_length,
        batch_size,
    ):
        super(Net, self).__init__()
        self.prot_model = prot_model
        self.prot_tokenizer = prot_tokenizer
        self.chemBERTa_model = chemBERTa_model
        self.chemBERTa_tokenizer = chemBERTa_tokenizer
        self.product_smiles_vocab = product_smiles_vocab
        self.max_seq_length = max_seq_length
        self.max_mol_length = max_mol_length
        self.batch_size = batch_size
        for param in self.prot_model.parameters():
            param.requires_grad = False
        for param in self.chemBERTa_model.parameters():
            param.requires_grad = False

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1024 + 384, 1024 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1024 + 384),
            torch.nn.Linear(1024 + 384, 1024 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1024 + 384),
        )
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(1024 + 384, 1024 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1024 + 384),
            torch.nn.Linear(1024 + 384, len(self.product_smiles_vocab)),
        )

        self.decoder2 = torch.nn.Sequential(
            torch.nn.Linear(1024 + 384, 1024 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1024 + 384),
            torch.nn.Linear(1024 + 384, len(self.product_smiles_vocab)),
        )

        self.decoder3 = torch.nn.Sequential(
            torch.nn.Linear(1024 + 384, 1024 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1024 + 384),
            torch.nn.Linear(1024 + 384, len(self.product_smiles_vocab)),
        )

    # 前向传播方法，传入酶序列和底物序列，返回预测的产物向量
    # Forward propagation method, input enzyme sequence and substrate sequence, return predicted product vector
    def forward(
        self,
        sequence_input_ids,
        sequence_attention_mask,
        substrate_input_ids,
        substrate_attention_mask,
    ):
        # 去除张量的第1个轴
        # Remove the first axis of the tensor
        sequence_input_ids = sequence_input_ids[:, 0, :]
        sequence_attention_mask = sequence_attention_mask[:, 0, :]
        substrate_input_ids = substrate_input_ids[:, 0, :]
        substrate_attention_mask = substrate_attention_mask[:, 0, :]
        # 对酶序列向量进行前向传播，得到酶序列的输出向量
        # Perform forward propagation on the enzyme sequence vector to obtain the output vector of the enzyme sequence
        with torch.no_grad():
            sequence_output = self.prot_model(
                input_ids=sequence_input_ids, attention_mask=sequence_attention_mask
            )

        sequence_embedding = sequence_output.last_hidden_state.detach().mean(axis=1)
        if (sequence_embedding is None) or (sequence_embedding == ""):
            sequence_embedding = torch.zeros(self.batch_size, 1024)
        # 对底物分子序列向量进行前向传播，得到底物分子序列的输出向量
        # Perform forward propagation on the substrate molecule sequence vector to obtain the output vector of the substrate molecule sequence
        with torch.no_grad():
            substrate_output = self.chemBERTa_model(
                input_ids=substrate_input_ids, attention_mask=substrate_attention_mask
            )

        substrate_embedding = substrate_output.last_hidden_state.detach().mean(axis=1)
        if (substrate_embedding is None) or (substrate_embedding == ""):
            substrate_embedding = torch.zeros(self.batch_size, 384)

        # 将酶序列和底物分子序列的输出向量拼接起来，作为DNN的输入向量
        # Concatenate the output vectors of enzyme sequence and substrate molecule sequence together as the input vector of DNN
        input_vector = torch.cat(
            (sequence_embedding, substrate_embedding), dim=1)

        # 通过decoder，得到预测的产物概率
        # Get the predicted product probability through the decoder
        predicted_vector1 = self.decoder1(self.encoder(input_vector))
        predicted_vector2 = self.decoder2(self.encoder(input_vector))
        predicted_vector3 = self.decoder3(self.encoder(input_vector))
        # 返回预测向量
        # Return prediction vector
        return predicted_vector1, predicted_vector2, predicted_vector3


# 定义模型NetESM类
# Define model NetESM class
class NetESM(torch.nn.Module):
    # 初始化方法，传入蛋白语言模型和化学分子模型的模型和分词器，定义DNN网络的encoder和decoder
    # Initialization method, pass in the model and word segmenter of the protein language model and chemical molecule model, and define the encoder and decoder of the DNN network
    def __init__(
        self,
        prot_model,
        prot_tokenizer,
        chemBERTa_model,
        chemBERTa_tokenizer,
        product_smiles_vocab,
        max_seq_length,
        max_mol_length,
        batch_size,
    ):
        super(NetESM, self).__init__()
        self.prot_model = prot_model
        self.prot_tokenizer = prot_tokenizer
        self.chemBERTa_model = chemBERTa_model
        self.chemBERTa_tokenizer = chemBERTa_tokenizer
        self.product_smiles_vocab = product_smiles_vocab
        self.max_seq_length = max_seq_length
        self.max_mol_length = max_mol_length
        self.batch_size = batch_size

        for param in self.prot_model.parameters():
            param.requires_grad = False
        for param in self.chemBERTa_model.parameters():
            param.requires_grad = False

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(1280 + 384, 1280 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1280 + 384),
            torch.nn.Linear(1280 + 384, 1280 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1280 + 384),
        )
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(1280 + 384, 1280 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1280 + 384),
            torch.nn.Linear(1280 + 384, len(self.product_smiles_vocab)),
        )

        self.decoder2 = torch.nn.Sequential(
            torch.nn.Linear(1280 + 384, 1280 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1280 + 384),
            torch.nn.Linear(1280 + 384, len(self.product_smiles_vocab)),
        )

        self.decoder3 = torch.nn.Sequential(
            torch.nn.Linear(1280 + 384, 1280 + 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.LayerNorm(1280 + 384),
            torch.nn.Linear(1280 + 384, len(self.product_smiles_vocab)),
        )

    # 前向传播方法，传入酶序列和底物序列，返回预测的产物向量
    # Forward propagation method, input enzyme sequence and substrate sequence, return predicted product vector
    def forward(
        self,
        sequence_input_ids,
        sequence_attention_mask,
        substrate_input_ids,
        substrate_attention_mask,
    ):
        # 去除张量的第1个轴
        # Remove the first axis of the tensor
        sequence_input_ids = sequence_input_ids[:, 0, :]
        sequence_attention_mask = sequence_attention_mask[:, 0, :]
        substrate_input_ids = substrate_input_ids[:, 0, :]
        substrate_attention_mask = substrate_attention_mask[:, 0, :]
        # 对酶序列向量进行前向传播，得到酶序列的输出向量
        # Perform forward propagation on the enzyme sequence vector to obtain the output vector of the enzyme sequence
        with torch.no_grad():
            sequence_output = self.prot_model(
                input_ids=sequence_input_ids, attention_mask=sequence_attention_mask
            )

        sequence_embedding = sequence_output.last_hidden_state.detach().mean(axis=1)
        if (sequence_embedding is None) or (sequence_embedding == ""):
            sequence_embedding = torch.zeros(self.batch_size, 1280)
        # 对底物分子序列向量进行前向传播，得到底物分子序列的输出向量
        # Perform forward propagation on the substrate molecule sequence vector to obtain the output vector of the substrate molecule sequence
        with torch.no_grad():
            substrate_output = self.chemBERTa_model(
                input_ids=substrate_input_ids, attention_mask=substrate_attention_mask
            )

        substrate_embedding = substrate_output.last_hidden_state.detach().mean(axis=1)
        if (substrate_embedding is None) or (substrate_embedding == ""):
            substrate_embedding = torch.zeros(self.batch_size, 384)

        # 将酶序列和底物分子序列的输出向量拼接起来，作为DNN的输入向量
        # Concatenate the output vectors of enzyme sequence and substrate molecule sequence together as the input vector of DNN
        input_vector = torch.cat(
            (sequence_embedding, substrate_embedding), dim=1)

        # 通过decoder，得到预测的产物概率
        # Get the predicted product probability through the decoder
        predicted_vector1 = self.decoder1(self.encoder(input_vector))
        predicted_vector2 = self.decoder2(self.encoder(input_vector))
        predicted_vector3 = self.decoder3(self.encoder(input_vector))
        # 返回预测向量
        # Return prediction vector
        return predicted_vector1, predicted_vector2, predicted_vector3