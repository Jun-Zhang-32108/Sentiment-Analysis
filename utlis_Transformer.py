import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Config
from collections import namedtuple
Config_Tuple = namedtuple('Config_Tuple', field_names="embed_dim,"
                   "hidden_dim,"
                   "num_embeddings,"
                   "num_max_positions,"
                   "num_heads,"
                   "num_layers,"
                   "dropout,"
                   "causal,"
                   "init_range,"
                   "device,"
                    "num_classes")

config = Config_Tuple(410, 2100, 28996, 256, 10, 16, 0.05,True, 0.02, device, 2)

# Model Definition
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    "Adopted from https://github.com/huggingface/naacl_transfer_learning_tutorial"

    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal):
        super().__init__()
        self.causal = causal
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithClfHead(nn.Module):
    "Adopted from https://github.com/huggingface/naacl_transfer_learning_tutorial"
    def __init__(self, config=config):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                       config.num_max_positions, config.num_heads, config.num_layers,
                                       config.dropout, causal=config.causal)
        
        self.classification_head = nn.Linear(config.embed_dim, config.num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, clf_tokens_mask, clf_labels=None, padding_mask=None):
        hidden_states = self.transformer(x, padding_mask)

        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)

        if clf_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(clf_logits.view(-1, clf_logits.size(-1)), clf_labels.view(-1))
            return clf_logits, loss
        return clf_logits

def predict(model, tokenizer, input="test"):
    "predict `input` with `model`"
    model.eval()
    tok = tokenizer.tokenize(input)
    ids = tokenizer.convert_tokens_to_ids(tok) + [tokenizer.vocab['[CLS]']]
    tensor = torch.tensor(ids, dtype=torch.long)
    tensor = tensor.to(device)
    tensor = tensor.reshape(1, -1)
    tensor_in = tensor.transpose(0, 1).contiguous() # [S, 1]
    logits = model(tensor_in,
                   clf_tokens_mask = (tensor_in == tokenizer.vocab['[CLS]']),
                   padding_mask = (tensor == tokenizer.vocab['[PAD]']))
    val, _ = torch.max(logits, 0)
    val = F.softmax(val, dim=0).detach().cpu().numpy()    
    return [(val.argmax(), val.max()),
            (val.argmin(), val.min())]


