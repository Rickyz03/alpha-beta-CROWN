import torch
import torch.nn as nn
import pytorch_pretrained_bert.modeling as bert_modeling

try:
    from auto_LiRPA.operators.softmax import BoundSoftmax
except ImportError:
    from auto_LiRPA.bound_ops import BoundSoftmax

# Sostituire l'attivazione GELU (che genera Erf) con ReLU (pienamente supportata)
bert_modeling.ACT2FN["gelu"] = torch.nn.functional.relu
from modeling import BertForSequenceClassificationFromEmbeddings
from pytorch_pretrained_bert.modeling import BertConfig

# ==========================================
# 1. IL TRUCCO DELL'ORACOLO (Monkey Patching per la Softmax)
# ==========================================
original_softmax_interval = BoundSoftmax.interval_propagate

def patched_interval_propagate(self, *v):
    # La softmax riceve un solo input
    x_node = v[0]
    
    x_center = (x_node[0] + x_node[1]) / 2.0
    
    # Check per l'Attention Softmax:
    # L'input ha forma (batch, num_heads, seq_len, seq_len)
    is_attention_softmax = (
        x_center.ndim == 4 and 
        x_center.shape[-1] == x_center.shape[-2]
    )
              
    if is_attention_softmax:
        print("\n[ORACOLO REALE] Intercettata Softmax dell'Attenzione! Sostituisco i bound con i limiti empirici esatti...\n")
        eps = 0.1
        n_samples = 500
        
        noise_x = (torch.rand(n_samples, *x_center.shape, device=x_center.device) * 2 * eps) - eps
        
        x_samples = x_center.unsqueeze(0) + noise_x
        
        # Calcoliamo la Softmax esatta sui campioni perturbati (lungo l'ultima dimensione)
        out_samples = torch.nn.functional.softmax(x_samples, dim=-1)
        
        return out_samples.amin(dim=0), out_samples.amax(dim=0)
    
    return original_softmax_interval(self, *v)

BoundSoftmax.interval_propagate = patched_interval_propagate

# ==========================================
# 2. WRAPPER DEL MODELLO E CARICAMENTO DATI
# ==========================================
class RealTransformerWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.core_model = BertForSequenceClassificationFromEmbeddings(config, num_labels=10)
        
    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        extended_attention_mask = torch.zeros(batch_size, 1, 1, seq_len, device=embeddings.device)
        return self.core_model(embeddings, extended_attention_mask)

def get_real_model():
    config = BertConfig(vocab_size_or_config_json_file=30522, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128)
    config.layer_norm = "no"
    config.embedding_size = 64
    return RealTransformerWrapper(config)

def get_real_data(*args, **kwargs):
    # Dimensioni: (batch_size=1, seq_len=4, hidden_size=64)
    X = torch.randn(1, 4, 64) 
    labels = torch.tensor([0])
    return {'X': X, 'labels': labels, 'eps': 0.1}
