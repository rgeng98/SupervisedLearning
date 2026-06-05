import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable

class Expert(nn.Module):
    """A single expert network. In an LLM or CV backbone, this is typically an MLP."""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    Sparsely-Gated Mixture of Experts Layer.
    Routes incoming tokens/features to the top-k most relevant experts.
    """
    def __init__(self, d_model: int, d_hidden: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.input_norm  = nn.LayerNorm(d_model)

        # 1. Instantiate the pool of independent experts
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(num_experts)])
        
        # 2. Gating network maps input features straight to expert scores
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original shape for tensor restoration at the end
        # Works perfectly for both 2D tabular/pooled data [batch, features] 
        # and 3D sequential text/vision data [batch, tokens/pixels, features]
        orig_shape = x.shape
        y = x
        x = x.view(-1, orig_shape[-1]) # Flatten to 2D matrix: [total_tokens, d_model]
        
        # Normalize inputs
        x = self.input_norm(x)

        # Step 1: Compute gate logits and keep only the top_k options
        gate_logits = self.gate(x) # [total_tokens, num_experts]
        topk_weights, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Step 2: Apply softmax over the chosen top-k routes to get routing probabilities
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        # Step 3: Initialize the final combined output buffer matrix
        final_output = torch.zeros_like(x)
        
        # Step 4: Route tokens to selected experts and combine results
        # We loop over the 'k' pathways rather than looping over all experts.
        # This keeps routing highly efficient when num_experts is large.
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]       # Which expert did each token choose?
            weight = topk_weights[:, i].unsqueeze(-1) # The scaling factor for that choice
            
            # Group tokens processing by individual expert assignment
            for core_expert_id in range(self.num_experts):
                # Mask out tokens that didn't choose this specific expert
                token_mask = (expert_idx == core_expert_id)
                if not token_mask.any():
                    continue
                    
                # Pass matching tokens through the expert, scale by gate weight, and accumulate
                expert_out = self.experts[core_expert_id](x[token_mask])
                final_output[token_mask] += weight[token_mask] * expert_out
                
        # Restore output back to original 2D or 3D sequence dimension mapping
        return final_output.view(*orig_shape) + y

class MoNE(MoELayer):
    def __init__(self, num_experts: int, top_k: int,  expert: Dict[str, int]):
        super().__init__(d_model=expert["d_model"], d_hidden=expert["d_hidden"], num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList([MoELayer(expert["d_model"], expert["d_hidden"], num_experts, top_k) for _ in range(num_experts)])
    
    

def factory(config: Dict[str, int | Dict[str, int]]) -> Callable:
    keys = config.keys()
    if "NestedMoE" in keys:
        return MoNE(config["num_experts"], config["top_k"], config["NestedMoE"])
    else:
        return MoELayer(config["d_model"], config["d_hidden"], config["num_experts"], config["top_k"])