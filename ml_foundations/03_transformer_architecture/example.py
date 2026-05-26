"""ml_foundations · 03 · Transformer architecture — runnable demo.

Builds scaled dot-product attention by hand, compares it against
PyTorch's `nn.MultiheadAttention` (they should agree to ~1e-6), runs a
forward pass on a real tokenised sentence, and saves the attention map
as a heatmap.

Run:
    uv sync --extra ml
    uv run python -m ml_foundations.03_transformer_architecture.example
"""

from __future__ import annotations

import math
from pathlib import Path

from shared import settings
from shared.pretty import console, section

OUT_DIR = settings.data_dir / "transformer"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _tokenise(text: str):
    """GPT-4o tokenizer via tiktoken — same one most lessons reference."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        ids = enc.encode(text)
        # Decode each id back to its display string so we can label the heatmap.
        labels = [enc.decode([i]) for i in ids]
        return ids, labels
    except ImportError:
        console.print("[yellow]Missing tiktoken. Run: uv sync --extra ml[/]")
        raise


def manual_attention(Q, K, V, mask=None):
    """Scaled dot-product attention — the six-line version.

    Q, K, V: (batch, n, d_k)
    Returns: (batch, n, d_k), attention weights (batch, n, n)
    """
    import torch
    import torch.nn.functional as F

    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)     # (B, n, n)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)                       # row-stochastic
    out = attn @ V                                          # (B, n, d_k)
    return out, attn


def multi_head_attention_manual(X, W_q, W_k, W_v, W_o, n_heads):
    """Multi-head self-attention written out so you can read it.

    X: (B, n, d_model)
    Returns: (B, n, d_model), attention weights (B, h, n, n)
    """
    import torch

    B, n, d_model = X.shape
    d_k = d_model // n_heads

    # Project + reshape into heads.
    def split(W):
        # (B, n, d_model) → (B, n, h, d_k) → (B, h, n, d_k)
        return (X @ W).view(B, n, n_heads, d_k).transpose(1, 2)

    Q, K, V = split(W_q), split(W_k), split(W_v)

    # Scaled dot-product per head — broadcast works fine.
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)        # (B, h, n, n)
    attn = torch.softmax(scores, dim=-1)
    head_out = attn @ V                                       # (B, h, n, d_k)

    # Concat heads.
    head_out = head_out.transpose(1, 2).contiguous().view(B, n, d_model)
    return head_out @ W_o, attn


def step1_show_tokens(text: str):
    section(f"STEP 1 · tokenise {text!r}")
    ids, labels = _tokenise(text)
    console.print(f"ids ({len(ids)}): {ids}")
    console.print(f"labels: {labels}")
    return ids, labels


def step2_compare_manual_vs_torch(d_model: int, n_heads: int, ids):
    section("STEP 2 · manual self-attention vs nn.MultiheadAttention")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        console.print("[yellow]Missing torch. Run: uv sync --extra ml[/]")
        return None, None

    torch.manual_seed(0)

    # Embed token ids → vectors (random init; this is the bottom layer of any transformer).
    vocab_size = 200_000     # > biggest GPT-4o id
    emb = nn.Embedding(vocab_size, d_model)
    X = emb(torch.tensor([ids]))                  # (1, n, d_model)

    # nn.MultiheadAttention with batch_first=True.
    mha = nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True)

    # Pull out its weight tensors so we can mirror them in the manual version.
    # `mha.in_proj_weight` stacks (W_q, W_k, W_v) row-wise; `mha.out_proj.weight`
    # is W_o. Both are stored as (d_model, d_model) — transpose for X @ W form.
    Wqkv = mha.in_proj_weight.detach().T          # (d, 3*d)
    W_q = Wqkv[:, :d_model]
    W_k = Wqkv[:, d_model:2 * d_model]
    W_v = Wqkv[:, 2 * d_model:]
    W_o = mha.out_proj.weight.detach().T          # (d, d)

    out_manual, attn_manual = multi_head_attention_manual(X, W_q, W_k, W_v, W_o, n_heads)
    out_torch, attn_torch = mha(X, X, X, need_weights=True, average_attn_weights=False)

    diff = (out_manual - out_torch).abs().max().item()
    console.print(f"max abs diff (manual vs nn.MultiheadAttention): {diff:.2e}")
    console.print(
        "[green]✓ matches[/]" if diff < 1e-4
        else "[red]✗ mismatch — check tensor layout[/]"
    )
    return out_manual, attn_manual


def step3_visualise_attention(attn, labels):
    section("STEP 3 · attention heatmap")
    try:
        import matplotlib.pyplot as plt
        import torch
    except ImportError:
        console.print("[yellow]Missing matplotlib. Skipping heatmap.[/]")
        return

    # attn: (B=1, h, n, n) — average over heads for display.
    avg = attn[0].mean(dim=0).detach().numpy()    # (n, n)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(avg, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l.strip() or "_" for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([l.strip() or "_" for l in labels])
    ax.set_xlabel("attended-to token (K)")
    ax.set_ylabel("attending token (Q)")
    ax.set_title("Mean attention across heads (random-init weights)")
    fig.colorbar(im, ax=ax)
    out = OUT_DIR / "attn_map.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    console.print(f"saved → {out}")


def step4_show_block(d_model: int, n_heads: int):
    section("STEP 4 · a full transformer block (pre-norm, with residuals)")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        return

    class TransformerBlock(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, h, bias=False, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, d),
            )

        def forward(self, x, attn_mask=None):
            # Pre-norm + residual around attention
            normed = self.ln1(x)
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
            x = x + attn_out
            # Pre-norm + residual around FFN
            x = x + self.ffn(self.ln2(x))
            return x

    torch.manual_seed(0)
    block = TransformerBlock(d_model, n_heads)
    n_params = sum(p.numel() for p in block.parameters())
    console.print(f"block params: {n_params:,}  (d={d_model}, h={n_heads}, FFN=4d)")
    breakdown = {
        "ln1": sum(p.numel() for p in block.ln1.parameters()),
        "attn (Q,K,V,O)": sum(p.numel() for p in block.attn.parameters()),
        "ln2": sum(p.numel() for p in block.ln2.parameters()),
        "ffn (4d)": sum(p.numel() for p in block.ffn.parameters()),
    }
    for k, v in breakdown.items():
        console.print(f"  {k:18}  {v:>10,}  ({v / n_params:.0%})")
    console.print("[dim]Notice: FFN dominates. ~2/3 of params live there in real models too.[/]")


def main() -> None:
    text = "The cat sat on the mat."
    d_model = 64
    n_heads = 4

    ids, labels = step1_show_tokens(text)
    _, attn = step2_compare_manual_vs_torch(d_model, n_heads, ids)
    if attn is not None:
        step3_visualise_attention(attn, labels)
    step4_show_block(d_model, n_heads)


if __name__ == "__main__":
    main()
