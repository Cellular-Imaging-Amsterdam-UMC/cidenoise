"""Equivalent PyTorch SDPA attention without an external FlashAttention build."""
import types
import torch.nn.functional as F


def normal_attention(self, q, k, v):
    q = q.view(*q.shape[:2], self.n_heads, -1).transpose(1, 2)
    k = k.view(*k.shape[:2], self.n_heads, -1).transpose(1, 2)
    v = v.view(*v.shape[:2], self.n_heads, -1).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
    out = out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
    return self.to_out(out)


def install(model):
    from .vendor.fluoresfm.unet_attention import CrossAttention
    for module in model.modules():
        if isinstance(module, CrossAttention):
            module.normal_attention = types.MethodType(normal_attention, module)
            module.flash = None
