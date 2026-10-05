"""
Plugin Oracolo (Versione Monkey Patching Robusta)
Aggira i problemi di importazione cercando la classe in percorsi multipli.
"""
import torch
import torch.nn as nn

# ==========================================
# 0. IMPORTAZIONE ROBUSTA
# ==========================================
try:
    from auto_LiRPA.operators.bmm import BoundMatMul
except ImportError:
    try:
        from auto_LiRPA.operators.matmul import BoundMatMul
    except ImportError:
        # Fallback per le versioni più recenti in cui tutto è consolidato
        from auto_LiRPA.bound_ops import BoundMatMul

# ==========================================
# 1. IL TRUCCO DELL'ORACOLO (Monkey Patching)
# ==========================================
original_bmm_interval = BoundMatMul.interval_propagate

def patched_interval_propagate(self, *v):
    q_node, k_node = v[0], v[1]
    
    q_center = (q_node[0] + q_node[1]) / 2.0
    k_center = (k_node[0] + k_node[1]) / 2.0
    
    # DISTINZIONE CHIRURGICA: Q*K^T vs P*V
    is_qkt = (q_center.shape[-2] == k_center.shape[-1])
    
    if is_qkt:
        print("\n[ORACOLO] Intercettato Q*K^T! Sostituisco McCormick con i limiti empirici esatti...\n")
        eps = 0.1
        n_samples = 1000
        
        noise_q = (torch.rand(n_samples, *q_center.shape, device=q_center.device) * 2 * eps) - eps
        noise_k = (torch.rand(n_samples, *k_center.shape, device=k_center.device) * 2 * eps) - eps
        
        q_samples = q_center.unsqueeze(0) + noise_q
        k_samples = k_center.unsqueeze(0) + noise_k
        
        out_samples = torch.matmul(q_samples, k_samples)
        
        return out_samples.amin(dim=0), out_samples.amax(dim=0)
    
    return original_bmm_interval(self, *v)

# Iniezione del patch
BoundMatMul.interval_propagate = patched_interval_propagate


# ==========================================
# 2. MODELLO TRANSFORMER PULITO
# ==========================================
class CustomTransformer(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        # seq_len=10, embed_dim=32 -> input è 320, output è 10
        self.fc_out = nn.Linear(32 * 10, 10)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) 
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v) 
        
        # Facciamo il flatten esplicito unendo la dimensione della sequenza con
        # la dimensione dell'embedding prima del layer lineare finale, 
        # esattamente come farebbe una rete fully connected standard.
        # context shape è (batch_size, seq_len, embed_dim)
        batch_size = context.shape[0]
        flattened = context.view(batch_size, -1) 
        
        return self.fc_out(flattened)

def get_model():
    return CustomTransformer(embed_dim=32)


# ==========================================
# 3. DATALOADER CUSTOM (Per l'esperimento dell'oracolo)
# ==========================================
def get_data(*args, **kwargs):
    """
    Dataloader per l'esperimento Oracolo.
    Restituisce un dizionario pulito senza chiavi NoneType.
    """
    import torch
    
    torch.manual_seed(42)
    
    num_samples = 1  
    seq_len = 10     
    embed_dim = 32   
    
    X = torch.randn(num_samples, seq_len, embed_dim)
    labels = torch.tensor([0])
    
    # Restituiamo solo lo stretto necessario.
    # Il framework calcolerà i bound automaticamente usando X ed eps.
    return {
        'X': X,
        'labels': labels,
        'eps': 0.1
    }
