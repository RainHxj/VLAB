from torch import nn
from model.txt_encoders.transformer import GELU
from torch.nn import LayerNorm as FusedLayerNorm

class BertHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()

        self.hidden_size = bert_model_embedding_weights.size(1)
        self.vocab_size = bert_model_embedding_weights.size(0)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = GELU()
        self.layernorm = FusedLayerNorm(self.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.decoder(sequence_output) 
        return prediction_scores

class DebertaV2Head(nn.Module):
    def __init__(self, config, embedding_weights):
        super().__init__()

        self.hidden_size = embedding_weights.size(1)
        self.vocab_size = embedding_weights.size(0)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = GELU()
        self.layernorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.decoder.weight = embedding_weights

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.decoder(sequence_output) 
        return prediction_scores


class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, cls_token):
        return self.linear(cls_token)

class MetaOPTHead(nn.Module):

    def __init__(self, config, model):
        super().__init__()
        self.cls = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.cls.weight = model.decoder.embed_tokens.weight
    def forward(self, x):
        return self.cls(x)