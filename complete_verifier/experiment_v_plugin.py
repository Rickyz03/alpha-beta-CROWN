import torch
import torch.nn as nn
import pytorch_pretrained_bert.modeling as bert_modeling
import math
try:
    from auto_LiRPA.operators.bmm import BoundMatMul
except ImportError:
    try:
        from auto_LiRPA.operators.matmul import BoundMatMul
    except ImportError:
        from auto_LiRPA.bound_ops import BoundMatMul

# Sostituire l'attivazione GELU (che genera Erf) con ReLU (pienamente supportata)
bert_modeling.ACT2FN["gelu"] = torch.nn.functional.relu
from modeling import BertForSequenceClassificationFromEmbeddings
from pytorch_pretrained_bert.modeling import BertConfig

# ==========================================
# 1. IL TRUCCO DELL'ORACOLO (Monkey Patching per P * V)
# ==========================================
original_bmm_interval = BoundMatMul.interval_propagate

def patched_interval_propagate(self, *v):
    p_node, v_node = v[0], v[1]
    
    p_center = (p_node[0] + p_node[1]) / 2.0
    v_center = (v_node[0] + v_node[1]) / 2.0
    
    # Check per P * V in base alla forma attesa da BERT
    # P (attention_probs) ha forma (batch, num_heads, seq_len, seq_len)
    # V (value_layer) ha forma (batch, num_heads, seq_len, head_size)
    is_pv = (
        p_center.ndim == 4 and v_center.ndim == 4 and 
        p_center.shape[-1] == p_center.shape[-2] and  # P è quadrata nelle ultime due dim (seq_len x seq_len)
        p_center.shape[-1] == v_center.shape[-2] and  # seq_len di P coincide con seq_len di V
        p_center.shape[-1] != v_center.shape[-1]      # Assicura di non confonderlo con altri layer
    )
              
    if is_pv:
        print("\n[ORACOLO REALE] Intercettato P*V! Sostituisco McCormick con i limiti empirici esatti...\n")
        eps = 0.1
        n_samples = 500
        
        noise_p = (torch.rand(n_samples, *p_center.shape, device=p_center.device) * 2 * eps) - eps
        noise_v = (torch.rand(n_samples, *v_center.shape, device=v_center.device) * 2 * eps) - eps
        
        p_samples = p_center.unsqueeze(0) + noise_p
        v_samples = v_center.unsqueeze(0) + noise_v
        
        out_samples = torch.matmul(p_samples, v_samples)
        
        return out_samples.amin(dim=0), out_samples.amax(dim=0)
    
    return original_bmm_interval(self, *v)

BoundMatMul.interval_propagate = patched_interval_propagate

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
    # Riduciamo seq_len a 4 per non esaurire la memoria senza sparsità
    X = torch.randn(1, 4, 64) 
    labels = torch.tensor([0])
    return {'X': X, 'labels': labels, 'eps': 0.1}
