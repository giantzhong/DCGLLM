"""
Standard Transformer model for EHR sequence prediction
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel


class DeltaDaysEmbedding(nn.Module):
    """
    Time interval embedding for visit-level delta_days
    """
    def __init__(
        self,
        embedding_dim: int,
        num_buckets: int = 64,
        max_days: int = 365 * 5,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_days = max_days
        self.embedding = nn.Embedding(num_buckets, embedding_dim)

    def bucketize(self, delta_days: torch.Tensor):
        """
        delta_days: (B, T), float
        log-scale bucketization
        """
        x = torch.clamp(delta_days, min=0.0, max=self.max_days)
        buckets = torch.floor(torch.log2(x + 1)).long()
        buckets = torch.clamp(buckets, 0, self.num_buckets - 1)
        return buckets

    def forward(self, delta_days: torch.Tensor):
        """
        delta_days: (B, T)
        """
        buckets = self.bucketize(delta_days)
        return self.embedding(buckets)


class TransformerModel(BaseModel):
    """
    Standard Transformer model for electronic health records.
    
    This implements a standard Transformer encoder architecture for processing
    sequential medical codes (diagnoses, procedures, drugs).
    
    Args:
        dataset: SampleEHRDataset
        feature_keys: List of feature keys to use (e.g., ["conditions"])
        label_key: Label key for prediction
        mode: Prediction mode ("binary", "multiclass", "multilabel")
        embedding_dim: Dimension of embeddings (default: 128)
        hidden_dim: Dimension of Transformer hidden states (default: 128)
        num_layers: Number of Transformer encoder layers (default: 1)
        num_heads: Number of attention heads (default: 4)
        train_dropout_rate: Dropout rate (default: 0.5)
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        train_dropout_rate: float = 0.5,
        **kwargs
    ):
        super(TransformerModel, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = train_dropout_rate
        
        # Feature tokenizers and embeddings
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        
        # Initialize feature-specific layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            
            # Validate input types
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Transformer only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "Transformer only supports 2-dim or 3-dim float and int as input types"
                )
            
            self.add_feature_transform_layer(feature_key, input_info)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            dropout=train_dropout_rate,
            max_len=5000
        )
        
        # Visit time embedding
        self.embed_visit_time = DeltaDaysEmbedding(
            embedding_dim=embedding_dim,
            num_buckets=64,
        )
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=train_dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Dropout
        self.dropout = nn.Dropout(train_dropout_rate)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Transformer model.
        
        Returns:
            patient_emb: Patient-level embeddings (batch_size, embedding_dim * num_features)
            patient_emb_all_step: All timestep embeddings (batch_size, seq_len, embedding_dim * num_features)
        """
        patient_emb = []
        patient_emb_all_step = []
        
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]
            
            # Process different input types
            if (dim_ == 2) and (type_ == str):
                # 2D string codes: (batch_size, seq_len)
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                mask = torch.any(x != 0, dim=2)
            
            elif (dim_ == 3) and (type_ == str):
                # 3D string codes: (batch_size, seq_len, num_codes)
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                x = self.embeddings[feature_key](x)
                x = torch.sum(x, dim=2)  # Aggregate over codes
                mask = torch.any(x != 0, dim=2)
            
            elif (dim_ == 2) and (type_ in [float, int]):
                # 2D numerical features
                x, mask = self.padding2d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = self.linear_layers[feature_key](x)
                mask = mask.bool().to(self.device)
            
            elif (dim_ == 3) and (type_ in [float, int]):
                # 3D numerical features
                x, mask = self.padding3d(kwargs[feature_key])
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)
            
            else:
                raise NotImplementedError

            # print(f"x.shape:{x.shape}")
            # Add visit time embedding if available
            if "delta_days" in kwargs and "visit_ids" in kwargs:
                td, _ = self.padding2d(kwargs["delta_days"])
                td = torch.tensor(td, dtype=torch.float, device=self.device)   # (B, T_visit)
                visit_time_emb = self.embed_visit_time(td)                     # (B, T_visit, D)
                visit_ids = torch.tensor(kwargs["visit_ids"], device=self.device)  # (B, L_token)
                # broadcast visit-level -> token-level
                visit_time_emb = visit_time_emb.gather(
                    1,
                    visit_ids.unsqueeze(-1).expand(-1, -1, visit_time_emb.size(-1))
                )  # (B, L_token, D)
                x = x + visit_time_emb
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Create attention mask (True = ignore, False = attend)
            # PyTorch Transformer uses additive mask: -inf for ignored positions
            attn_mask = ~mask  # Invert: True where padding
            attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf'))
            attn_mask = attn_mask.masked_fill(~attn_mask.bool(), float(0.0))
            
            # Pass through Transformer encoder
            transformer_out = self.transformer_encoder(
                x,
                src_key_padding_mask=~mask  # True = ignore padding
            )
            
            # Get final representation
            # Use the last valid token for each sequence
            lengths = mask.sum(dim=1).long() - 1  # Last valid position
            batch_indices = torch.arange(x.size(0), device=self.device)
            final_hidden = transformer_out[batch_indices, lengths, :]
            
            patient_emb.append(final_hidden)
            patient_emb_all_step.append(transformer_out)
        
        # Concatenate features
        patient_emb = torch.cat(patient_emb, dim=1)
        patient_emb_all_step = torch.cat(patient_emb_all_step, dim=2)
        
        return patient_emb, patient_emb_all_step


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    
    Adds positional information to input embeddings using sinusoidal functions.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (max_len, 1, d_model)
        # We need to add pe[:seq_len] to x
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)
        return self.dropout(x)

