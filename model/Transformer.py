import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.W_k = nn.Parameter(torch.rand([1, d_model]))
        self.register_buffer('pe', pe)

    def forward(self, x, timestamps=None):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            #>>> output = pos_encoder(x)
        """
        if timestamps is not None:
            # timestamps [seq_length]
            timestamps = timestamps.unsqueeze(dim=1)
            # timestamps [seq_length, 1]
            time_tensor = torch.mul(timestamps, self.W_k)
            time_tensor = torch.sin(time_tensor)
            # timestamps [seq_length, dim]
            time_tensor = time_tensor.unsqueeze(0).transpose(0, 1) # [seq_length, 1, dim]
            pe = self.pe[:x.size(0), :]
            # pe [seq_length,1, dim]
            return x + time_tensor + pe

        x = x + self.pe[:x.size(0), :]
        #print(x.shape)
        return self.dropout(x)

class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

    def forward(self, x, timestamps):
        # timestamps [seq_length]
        result = timestamps.unsqueeze(dim=-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return x + result




class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        #self.decoder = nn.Linear(ninp, ntoken)

        #self.linear = nn.Linear(ninp, 1)

        # parameter for the weight of time difference
        #self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        #self.beta = nn.Parameter(torch.tensor(1.0))
        #self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, src, timestamps=None, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        #src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src, timestamps)
        output = self.transformer_encoder(src, self.src_mask)
        return output
        #output = self.decoder(output)
        #return F.log_softmax(output, dim=-1)


if __name__ == '__main__':
    model = TransformerModel(ninp=768, nhead=8, nhid=768, nlayers=6, dropout=0.5)


    src = torch.rand([10, 1, 768])
    res = model(src)
    #res = model.transformer_encoder(src)


