import math
import torch
import torch.nn.functional as F
from torch import nn, einsum 

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape

from rotary_embedding_torch import RotaryEmbedding

from model_structure.diff_utils.model_utils import *

from random import sample

class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim, 
        depth,
        dim_in_out=None,
        cross_attn=True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True, 
        attn_dropout = 0.,
        ff_dropout = 0.3,
        final_proj = True, 
        normformer = False,
        rotary_emb = True,
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)

        self.use_same_dims = False
        point_feature_dim = 768

        self.simply_layers = nn.ModuleList([])

        self.simply_layers.append(nn.ModuleList([
            Attention(dim=dim, out_dim=dim, causal=True, dim_head=dim_head, heads=heads, rotary_emb=rotary_emb),
            Attention(dim=dim, kv_dim=point_feature_dim, causal=True, dim_head=dim_head, heads=heads,
                      dropout=attn_dropout, rotary_emb=rotary_emb_cross),
            FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer)
        ]))
        for _ in range(depth):
            self.simply_layers.append(nn.ModuleList([
                Attention(dim=dim, causal=True, dim_head=dim_head, heads=heads, rotary_emb=rotary_emb),
                Attention(dim=dim, kv_dim=point_feature_dim, causal=True, dim_head=dim_head, heads=heads,
                          dropout=attn_dropout, rotary_emb=rotary_emb_cross),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer)
            ]))
        self.simply_layers.append(nn.ModuleList([
            Attention(dim=dim, out_dim=dim, causal=True, dim_head=dim_head, heads=heads, rotary_emb=rotary_emb),
            Attention(dim=dim, kv_dim=point_feature_dim, out_dim=dim_in_out, causal=True, dim_head=dim_head,
                      heads=heads, dropout=attn_dropout, rotary_emb=rotary_emb_cross),
            FeedForward(dim=dim_in_out, out_dim=dim_in_out, mult=ff_mult, dropout=ff_dropout,
                        post_activation_norm=normformer)
        ]))

        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()
        self.project_out = nn.Linear(dim_in_out, dim_in_out, bias = False) if final_proj else nn.Identity()

        self.cross_attn = cross_attn



    def forward(self, x, time_emb=None, context=None):

        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)


        for idx, (self_attn, cross_attn, ff) in enumerate(self.simply_layers):

            if (idx == 0 or idx == 5):
                x = self_attn(x, attn_bias=attn_bias)
                x = cross_attn(x, context=context)
            else:
                x = self_attn(x, attn_bias=attn_bias) + x
                x = cross_attn(x, context=context) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

class DiffusionNet(nn.Module):

    def __init__(
        self,
        dim,
        depth,
        dim_in_out=None,
        num_timesteps = None,
        num_time_embeds = 1,
        cond = True,
        **kwargs
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.depth = depth
        self.cond = cond
        self.cross_attn = True
        self.cond_dropout = kwargs.get('cond_dropout', False)
        self.point_feature_dim = kwargs.get('point_feature_dim', dim)

        self.dim_in_out = default(dim_in_out, dim)

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )
        self.learned_query = nn.Parameter(torch.randn(self.dim_in_out))
        self.causal_transformer = CausalTransformer(dim=dim, depth=self.depth,dim_in_out=self.dim_in_out, **kwargs)


    def forward(
        self,
        data, 
        diffusion_timesteps,
        cond_feature,
        pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling
    ):
            
        batch, dim, device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds(diffusion_timesteps)

        data = data.unsqueeze(1)

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        model_inputs = [time_embed, data, learned_queries]

        tokens = torch.cat(model_inputs, dim = 1)

        tokens = self.causal_transformer(tokens, context=cond_feature)
        pred = tokens[..., -1, :]

        return pred

