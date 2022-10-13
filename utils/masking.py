import torch

# Obsismc
# mask is to make sure that every timestamp is only dependent on its previous timestamps
# it is used in the first attention module of decoder

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

# Obsismc: mask ProbAttention via index
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)  # Obsismc: triu(1) == triu(,diagonal=1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        # Obsismc: cannot directly get 'top_u' tensors rather than using index?
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask