import torch
import torch.nn as nn
import pytorch_pretrained_bert.modeling as bert_modeling
import auto_LiRPA
import math

bert_modeling.ACT2FN["gelu"] = torch.nn.functional.relu
from modeling import BertForSequenceClassificationFromEmbeddings
from pytorch_pretrained_bert.modeling import BertConfig

class RealTransformerWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.core_model = BertForSequenceClassificationFromEmbeddings(config, num_labels=2)
        
    def forward(self, embeddings):
        batch_size, seq_len, _ = embeddings.shape
        extended_attention_mask = torch.zeros(batch_size, 1, 1, seq_len, device=embeddings.device)
        return self.core_model(embeddings, extended_attention_mask)

def get_real_model():
    config = BertConfig(vocab_size_or_config_json_file=7145, hidden_size=64, num_hidden_layers=1, num_attention_heads=4, intermediate_size=128)
    config.layer_norm = "no_var"
    config.embedding_size = 64
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    
    core_model = BertForSequenceClassificationFromEmbeddings(config, num_labels=2)
    dummy_embed = torch.zeros(1, 32, 64) 
    dummy_mask = torch.zeros(1, 1, 1, 32)
    temp_bounded = auto_LiRPA.BoundedModule(core_model, (dummy_embed, dummy_mask), device="cpu")
    ckpt = torch.load("Transformer-ckpt_11", map_location="cpu")
    temp_bounded.load_state_dict(ckpt['state_dict_model_from_embeddings'], strict=False)
    
    wrapper = RealTransformerWrapper(config)
    wrapper.core_model = core_model
    
    # SALVIAMO IL MODELLO IN UNA VARIABILE GLOBALE
    global GLOBAL_CORE_MODEL
    GLOBAL_CORE_MODEL = core_model
    
    return wrapper

def get_real_data(*args, **kwargs):
    config = BertConfig(vocab_size_or_config_json_file=7145, hidden_size=64, num_hidden_layers=1, num_attention_heads=4, intermediate_size=128)
    config.embedding_size = 64
    from modeling import BertEmbeddings
    embeddings_layer = BertEmbeddings(config)
    ckpt = torch.load("Transformer-ckpt_11", map_location="cpu")
    embeddings_layer.load_state_dict(ckpt['state_dict_embeddings'], strict=False)
    
    input_ids = torch.tensor([[1, 200, 300, 2]])
    with torch.no_grad():
        X = embeddings_layer(input_ids)
    labels = torch.tensor([0])
    return {'X': X, 'labels': labels, 'eps': 0.1}

original_compute_bounds = auto_LiRPA.BoundedModule.compute_bounds

def patched_compute_bounds(self, x=None, aux=None, C=None, method='backward', bound_lower=True, bound_upper=True, return_A=False,
                           interm_bounds=None, reference_bounds=None, **kwargs):
    
    if method != 'IBP':
        target_node = None
        for node in self.nodes():
            if isinstance(node, auto_LiRPA.operators.BoundSoftmax):
                x_shape = node.inputs[0].output_shape
                if len(x_shape) == 4 and x_shape[-1] == x_shape[-2]:
                    target_node = node.name
                    break
        
        if target_node:
            if not hasattr(self, '_cached_oracle_bounds'):
                print(f"\n[ORACOLO REALE] Calcolo e inietto limiti empirici esatti per SOFTMAX (nodo {target_node}) tramite reference_bounds...\n")
                eps = 0.1
                embeddings = x[0].data
                n_samples = 500
                noise = (torch.rand(n_samples, *embeddings.shape[1:], device=embeddings.device) * 2 * eps) - eps
                x_samples = embeddings.repeat(n_samples, 1, 1) + noise
                
                with torch.no_grad():
                    # RECUPERIAMO IL MODELLO DALLA VARIABILE GLOBALE
                    global GLOBAL_CORE_MODEL
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
                    probs = torch.nn.functional.softmax(scores, dim=-1)
                    
                    lb = probs.amin(dim=0, keepdim=True)
                    ub = probs.amax(dim=0, keepdim=True)
                
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