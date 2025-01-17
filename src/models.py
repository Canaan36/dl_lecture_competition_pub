import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        num_subjects: int, #new
        hid_dim: int = 128
        #hid_dim: int = 256  # 增加隐藏层维度
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            #ConvBlock(hid_dim, hid_dim),  # 增加卷积层
        )

        self.subject_embedding = nn.Embedding(num_subjects, hid_dim)#new
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            #nn.Linear(hid_dim, num_classes),
            nn.Linear(hid_dim*2, num_classes),#new
        )

    #def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        subject_emb = self.subject_embedding(subject_idx)
        subject_emb = subject_emb.unsqueeze(2).expand(-1, -1, X.size(2))  # Expand to match the sequence length
        X = torch.cat((X, subject_emb), dim=1)
        return self.head(X)
        
    #def load_pretrained_weights(self, weight_path: str) -> None:
    #    pretrained_weights = torch.load(weight_path)
    #    self.load_state_dict(pretrained_weights, strict=False)  # 严格模式为False，允许部分权重不匹配
    
    def load_pretrained_weights(self, weight_path: str) -> None:
        pretrained_weights = torch.load(weight_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
        #p_drop: float = 0.2,  # 增加 Dropout 概率
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)