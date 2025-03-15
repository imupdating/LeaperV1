# 这个文件定义模型的结构
import torch.nn as nn
import torch, math
import torch.nn.functional as F
import Ltokenizer

# 为了.make_data()函数的兼容性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 一些关键层
class LeaperBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, output_dim, dropout=0.1, use_residual=True):
        """
        初始化一个LeaperBlock模块。
        
        参数:
        - d_model: 输入/输出的特征维度
        - nhead: 多头注意力机制的头数
        - dim_feedforward: 前馈网络的隐藏层维度
        - output_dim: 输出维度
        - dropout: Dropout 概率
        - use_residual: 是否使用残差连接
        """
        super().__init__()
        self.use_residual = use_residual
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        
        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final Linear Layer
        self.final = nn.Linear(d_model, output_dim)
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        前向传播函数。
        
        参数:
        - x: 输入张量，形状为 (batch_size, seq_len, d_model)
        - src_mask: 注意力掩码
        - src_key_padding_mask: 填充掩码
        
        返回:
        - 输出张量，形状为 (batch_size, seq_len, output_dim)
        """
        # Step 1: Multi-Head Self-Attention
        x = self._apply_attention(x, src_mask, src_key_padding_mask)
        
        # Step 2: Feed-Forward Network
        x = self._apply_ffn(x)
        
        # Step 3: Final Linear Projection
        x = self.final(x)
        return x

    def _apply_attention(self, x, src_mask=None, src_key_padding_mask=None):
        """
        应用多头自注意力机制。
        
        参数:
        - x: 输入张量
        - src_mask: 注意力掩码
        - src_key_padding_mask: 填充掩码
        
        返回:
        - 经过注意力机制处理后的张量
        """
        residual = x
        x = self.norm1(x)
        x, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = residual + self.dropout(x) if self.use_residual else self.dropout(x)
        return x

    def _apply_ffn(self, x):
        """
        应用前馈神经网络。
        
        参数:
        - x: 输入张量
        
        返回:
        - 经过前馈网络处理后的张量
        """
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x) if self.use_residual else self.dropout(x)
        return x

# 主要的模型
class LeaperV1(nn.Module):
    # 参数说明
    # dict_size: 词典的大小
    def __init__(self, max_len, vocab_size, n_head, layer_num, d_model, dim_feedforward):
        # 调用父类的构造函数
        super(LeaperV1, self).__init__()
        self.d_model = d_model
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码(最大生成长度为max_len，只取决于这里)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        # 多层Block，每一层都需要掩码，因为每一层都需要屏蔽无效信息，并行计算时，如果每一层不屏蔽的话，有可能会看到未来信息
        self.blocks = nn.ModuleList([LeaperBlock(d_model, n_head, dim_feedforward, d_model) for _ in range(layer_num)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.output_layer(x)
    
    def generate_mask(self, sz):
        # 生成因果掩码
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

if __name__ == '__main__':
    word2id, id2word = Ltokenizer.get_dict_from_file("tokens.json")
    # 测试
    model = LeaperV1(100, len(word2id), 4, 2, 20, 20)
    print(model)
    # 生成一个含有一百字的随机输入
    x = torch.randint(0, len(word2id), (1, 100))
    y = model(x)
    print(x.shape, y.shape)