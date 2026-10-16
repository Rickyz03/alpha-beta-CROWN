import torch
import torch.nn as nn
import pytorch_pretrained_bert.modeling as bert_modeling
import auto_LiRPA
import math

bert_modeling.ACT2FN["gelu"] = torch.nn.functional.relu
from pytorch_pretrained_bert.modeling import BertConfig
from modeling import BertForSequenceClassificationFromEmbeddings

SANITY_CHECK = False  # <--- METTI A True PER IL TEST CHIESTO DAL PROF

def remove_dropout(m):
    """Eradica il Dropout per garantire la verifica deterministica"""
    for name, child in m.named_children():
        if isinstance(child, nn.Dropout):
            setattr(m, name, nn.Identity())
        else:
            remove_dropout(child)

class RealTransformerWrapper(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        extended_attention_mask = torch.zeros(batch_size, 1, 1, seq_len, device=embeddings.device)
        return self.core(embeddings, extended_attention_mask)

def get_real_model():
    config = BertConfig(vocab_size_or_config_json_file=7145, hidden_size=64, num_hidden_layers=1, num_attention_heads=4, intermediate_size=128)
    config.layer_norm = "no_var"
    config.embedding_size = 64
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    
    core_model = BertForSequenceClassificationFromEmbeddings(config, num_labels=2)
    remove_dropout(core_model)
    
    dummy_embed = torch.zeros(1, 32, 64) 
    dummy_mask = torch.zeros(1, 1, 1, 32)
    temp_bounded = auto_LiRPA.BoundedModule(core_model, (dummy_embed, dummy_mask), device="cpu")
    
    ckpt = torch.load("Transformer-ckpt_11", map_location="cpu")
    temp_bounded.load_state_dict(ckpt['state_dict_model_from_embeddings'], strict=False)
    
    wrapper = RealTransformerWrapper(core_model)
    global GLOBAL_CORE_MODEL
    GLOBAL_CORE_MODEL = core_model
    return wrapper

def get_real_data(*args, **kwargs):
    import sys
    import os

    # 1. Calcoliamo il salto esatto: saliamo di due cartelle e scendiamo in auto_LiRPA
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(current_dir, "..", "auto_LiRPA", "examples", "language"))

    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    # 2. Rubiamo temporaneamente il data_utils locale dalla cache di Python
    local_data_utils = sys.modules.pop('data_utils', None)

    # 3. Ora Python è costretto a leggere il file dalla cartella target_dir!
    from data_utils import load_data
    from Transformer.Transformer import Transformer

    # 4. Restituiamo il data_utils originale per non rompere il resto del codice di abcrown
    if local_data_utils is not None:
        sys.modules['data_utils'] = local_data_utils
    
    class DummyArgs:
        max_sent_length = 32; drop_unk = False; num_classes = 2; device = "cpu"
        lr = 1e-4; dir = "dummy"; min_word_freq = 2; num_layers = 1
        embedding_size = 64; hidden_size = 64; intermediate_size = 128
        hidden_act = "relu"; num_attention_heads = 4; layer_norm = "no_var"
        dropout = 0.0; load = None
        
    args = DummyArgs()
    _, data_train, _, data_test = load_data('sst')
    transformer = Transformer(args, data_train)
    
    ckpt = torch.load("Transformer-ckpt_11", map_location="cpu")
    transformer.model.embeddings.load_state_dict(ckpt['state_dict_embeddings'])
    
    # Il nostro Paziente Zero: Indice 64
    batch = [data_test[64]]
    embeddings, _, _, label_ids = transformer.get_input(batch)
    
    return {'X': embeddings.detach(), 'labels': label_ids, 'eps': 0.005}

# INIEZIONE DINAMICA SUL NODO PRE-SOFTMAX (/input-3)
original_compute_bounds = auto_LiRPA.BoundedModule.compute_bounds

def patched_compute_bounds(self, x=None, aux=None, C=None, method='backward', bound_lower=True, bound_upper=True, return_A=False, interm_bounds=None, reference_bounds=None, **kwargs):
    
    if method != 'IBP':
        target_node = None
        for node in self.nodes():
            if isinstance(node, auto_LiRPA.operators.BoundSoftmax):
                target_node = node.inputs[0].name  # Catturiamo dinamicamente l'input della Softmax
                break
        
        if target_node:
            if not hasattr(self, '_cached_oracle_bounds'):
                if SANITY_CHECK:
                    print(f"\n[SANITY CHECK] Inietto limiti FINTI e LARGHISSIMI [-1000, 1000] su {target_node}...")
                    print("[SANITY CHECK] Quei limiti non sono gestibili in memoria, vado di [-15, 15] per il test richiesto dal prof.")
                    dummy_shape = self[target_node].output_shape
                    lb = torch.ones(dummy_shape, device=x[0].device) * -15.0
                    ub = torch.ones(dummy_shape, device=x[0].device) * 15.0
                else:
                    print(f"\n[ORACOLO REALE] Calcolo limiti empirici perfetti per PRE-SOFTMAX (Nodo {target_node})")
                    eps = 0.005
                    embeddings = x[0].data
                    n_samples = 500
                    noise = (torch.rand(n_samples, *embeddings.shape[1:], device=embeddings.device) * 2 * eps) - eps
                    x_samples = embeddings.repeat(n_samples, 1, 1) + noise
                    
                    with torch.no_grad():
                        core_m = GLOBAL_CORE_MODEL
                        h = core_m.linear_in(x_samples)
                        if core_m.layer_norm != "no":
                            h = core_m.LayerNorm(h)
                        
                        self_attn = core_m.bert.encoder.layer[0].attention.self
                        q = self_attn.query(h)
                        k = self_attn.key(h)
                        
                        new_shape = q.size()[:-1] + (self_attn.num_attention_heads, self_attn.attention_head_size)
                        q = q.view(*new_shape).permute(0, 2, 1, 3)
                        k = k.view(*new_shape).permute(0, 2, 1, 3)
                        
                        scores = torch.matmul(q, k.transpose(-1, -2))
                        scores = scores / math.sqrt(self_attn.attention_head_size)
                        
                        mask = torch.zeros(1, 1, 1, embeddings.shape[1], device=embeddings.device)
                        scores = scores + mask
                        
                        lb = scores.amin(dim=0, keepdim=True)
                        ub = scores.amax(dim=0, keepdim=True)
                
                self._cached_oracle_bounds = (lb, ub)
            
            lb, ub = self._cached_oracle_bounds
            if reference_bounds is None: reference_bounds = {}
            else: reference_bounds = reference_bounds.copy()
            reference_bounds[target_node] = (lb, ub)
            if interm_bounds is None: interm_bounds = {}
            else: interm_bounds = interm_bounds.copy()
            interm_bounds[target_node] = (lb, ub)
            
    return original_compute_bounds(self, x=x, aux=aux, C=C, method=method, bound_lower=bound_lower, bound_upper=bound_upper, return_A=return_A, interm_bounds=interm_bounds, reference_bounds=reference_bounds, **kwargs)

auto_LiRPA.BoundedModule.compute_bounds = patched_compute_bounds
