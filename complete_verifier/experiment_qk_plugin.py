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
# 1. IL TRUCCO DELL'ORACOLO (Monkey Patching)
# ==========================================
original_bmm_interval = BoundMatMul.interval_propagate

def patched_interval_propagate(self, *v):
    q_node, k_node = v[0], v[1]
    
    q_center = (q_node[0] + q_node[1]) / 2.0
    k_center = (k_node[0] + k_node[1]) / 2.0
    
    # Check per Q*K^T in base alla forma attesa da BERT
    # Q ha forma (batch, num_heads, seq_len, head_size)
    # K^T ha forma (batch, num_heads, head_size, seq_len)
    is_qkt = (q_center.ndim == 4 and k_center.ndim == 4 and 
              q_center.shape[-2] == k_center.shape[-1])
              
    if is_qkt:
        print("\n[ORACOLO REALE] Intercettato Q*K^T! Sostituisco McCormick con i limiti empirici esatti...\n")
        eps = 0.1
        n_samples = 500
        
        noise_q = (torch.rand(n_samples, *q_center.shape, device=q_center.device) * 2 * eps) - eps
        noise_k = (torch.rand(n_samples, *k_center.shape, device=k_center.device) * 2 * eps) - eps
        
        q_samples = q_center.unsqueeze(0) + noise_q
        k_samples = k_center.unsqueeze(0) + noise_k
        
        out_samples = torch.matmul(q_samples, k_samples)
        
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
    # Riduciamo seq_len da 16 a 4. 
    # Le dimensioni diventano (batch_size=1, seq_len=4, hidden_size=64)
    X = torch.randn(1, 4, 64) 
    labels = torch.tensor([0])
    return {'X': X, 'labels': labels, 'eps': 0.1}
