import torch
from torch import nn
from einops import rearrange
from typing import List


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, q, **kwargs):
        return self.fn(self.norm(x), q, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, q=None):  # q is passed only for easier code
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, q):
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = rearrange(self.to_q(q), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, q):
        for attn, ff in self.layers:
            x = attn(x, q) + x
            x = ff(x, q) + x
        return x


class CascadedCrossModalTransformer(nn.Module):
    def __init__(
        self,
        num_outputs,
        num_patches,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.20,
        regression=False,
        modalities: List[str] | None = None,
    ):
        super().__init__()
        self.modalities = list(modalities or ["text_es", "text_en", "audio"])
        if not self.modalities:
            raise ValueError("CCMT necesita cel putin o modalitate")
        if num_patches % len(self.modalities) != 0:
            raise ValueError("The number of patches must be equal for all selected modalities")

        self.ppm = num_patches // len(self.modalities)
        self.pos_embeddings = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, self.ppm, dim))
            for modality in self.modalities
        })
        self.cross_transformers = nn.ModuleList([
            Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
            for _ in range(max(len(self.modalities) - 1, 0))
        ])


        self.regression = regression
        if not regression:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_outputs),
                nn.Softmax(dim=-1)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_outputs)
            )

    def forward(self, x):
        token_chunks = []
        for index, modality in enumerate(self.modalities):
            start = index * self.ppm
            end = (index + 1) * self.ppm
            token_chunks.append(x[:, start:end] + self.pos_embeddings[modality])

        fused_tokens = token_chunks[0]
        for transformer, query_tokens in zip(self.cross_transformers, token_chunks[1:]):
            fused_tokens = transformer(fused_tokens, query_tokens)

        x = fused_tokens[:, 0]
        return self.mlp_head(x)


if __name__ == '__main__':
    # Usage example
    model = CascadedCrossModalTransformer(
        num_outputs=3,
        num_patches=300,
        dim=1024,
        depth=6,
        heads=6,
        mlp_dim=128,
        modalities=["text_en", "text_es", "audio"],
    )
    result = model(torch.zeros((10, 300, 1024)))  # batch x tokens x dim_token
    print(result.shape)