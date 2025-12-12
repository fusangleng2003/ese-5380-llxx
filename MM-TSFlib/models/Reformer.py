import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding

class CrossAttention(nn.Module):
    def __init__(self, d_model, text_dim, seq_len, num_heads=4, dropout=0.1):
        super().__init__()
        
        # --- 1. Attention 部分 ---
        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(text_dim)

        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(text_dim, d_model)
        self.proj_v = nn.Linear(text_dim, d_model)
        
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
  
        self.gate_param = nn.Parameter(torch.full((1, seq_len, 1), 0.0))

    def forward(self, x_query, x_key_value):
        original_query = x_query
        
        # --- Step 1: Cross Attention ---
        q = self.proj_q(self.layer_norm_q(x_query))        
        k = self.proj_k(self.layer_norm_k(x_key_value))    
        v = self.proj_v(self.layer_norm_k(x_key_value))    
        
        attn_out, _ = self.attention(query=q, key=k, value=v)
        
        # Add & Norm
        x = x_query + self.dropout1(attn_out)
        x = self.norm1(x)

        # --- Step 2: FFN (Feed Forward) ---
        ffn_out = self.ffn(x)
        
        # Add & Norm
        x = x + ffn_out
        x = self.norm2(x)

        # --- Step 3: Gating Fusion ---
        alpha = torch.sigmoid(self.gate_param)
        final_output = (1 - alpha) * original_query + alpha * x
        
        return final_output

class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        """
        bucket_size: int, 
        n_hashes: int, 
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads,
                                  bucket_size=bucket_size, n_hashes=n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.cross_fusion = CrossAttention(
            d_model=configs.d_model,    
            text_dim=configs.d_model,   
            seq_len=configs.seq_len + configs.pred_len,
            num_heads=4,                
            dropout=0.3
        )

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        else:
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_emb = None):
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        #dec_out = self.projection(enc_out)

        if text_emb is not None:
            enc_out = self.cross_fusion(enc_out, text_emb)

        dec_out = self.projection(enc_out)

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        enc_out, attns = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        return enc_out  # [B, L, D]

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]

        enc_out, attns = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        return enc_out  # [B, L, D]

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_emb = None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_emb)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
