import torch

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 tensor of shape (batch, seq_q, d_v).
    Q: (batch, seq_q, d_k)
    K: (batch, seq_k, d_k)
    V: (batch, seq_k, d_v)
    """
    d_k = Q.shape[-1] 
    scores = Q @ K.transpose(-2, -1) * d_k ** -0.5
    seq_q, seq_k = scores.shape[-2], scores.shape[-1]
    causal_mask = torch.ones(
            seq_q, seq_k, dtype=torch.bool
        ).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V