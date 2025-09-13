import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    """
    Fused multi-head causal self-attention (Transformer-style).

    This module projects the input once into Q/K/V of size `d_out`, splits them
    into `num_heads` heads of size `head_dim = d_out // num_heads`, applies a
    **causal mask** (no future attention), computes attention per head, and then
    concatenates and linearly projects back to `d_out`.

    Args:
        d_in (int): Input embedding size.
        d_out (int): Total output size across all heads (also the model size).
        context_length (int): Maximum supported sequence length (mask size).
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability on attention weights.
        qkv_bias (bool, optional): Whether to include bias in Q/K/V projections.

    Shapes:
        Input:  x -> (B, T, d_in)
        Output: y -> (B, T, d_out)

        where:
            B = batch size, T = sequence length.
    """

    def __init__(self, d_in, d_out, 
                 context_length, num_heads,
                 dropout=0.0, qkv_bias=False):
        super().__init__()

        # Ensure the total output dimension splits evenly across heads.
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # Save basic hyperparameters.
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # per-head feature dimension

        # Single Q/K/V projections mapping d_in -> d_out (total across all heads).
        # (Corrected: nn.Linear, not nn.linear)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Register a (context_length x context_length) upper-triangular mask.
        # mask[i, j] = True when j > i (i.e., "future" positions to be masked).
        # (Fixed parentheses and made it boolean at creation.)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

        # Dropout applied to attention weights after softmax (not to scores).
        self.dropout = nn.Dropout(dropout)

        # Final linear projection after concatenating heads: (B, T, d_out) -> (B, T, d_out).
        self.out_proj = nn.Linear(d_out, d_out)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, T, d_in)
        Returns:
            torch.Tensor: (B, T, d_out)
        """
        # Unpack batch (B), time/sequence length (T), and channels (C=d_in).
        B, T, C = x.shape
        
        # Project inputs to fused Q/K/V of shape (B, T, d_out).
        queries = self.W_query(x)  # (B, T, d_out)
        keys    = self.W_key(x)    # (B, T, d_out)
        values  = self.W_value(x)  # (B, T, d_out)
        
        # Reshape to split heads: (B, T, d_out) -> (B, T, num_heads, head_dim)
        queries = queries.view(B, T, self.num_heads, self.head_dim)
        keys    = keys.view(B, T, self.num_heads, self.head_dim)
        values  = values.view(B, T, self.num_heads, self.head_dim)
        
        # Move heads before time for batched attention: (B, num_heads, T, head_dim)
        queries = queries.transpose(1, 2)  # (B, H, T, Hd)
        keys    = keys.transpose(1, 2)     # (B, H, T, Hd)
        values  = values.transpose(1, 2)   # (B, H, T, Hd)
        
        # Compute attention scores per head: (B, H, T, Hd) @ (B, H, Hd, T) -> (B, H, T, T)
        att_scores = queries @ keys.transpose(2, 3)

        # Scale by sqrt(head_dim) (the size of each key/query vector).
        att_scores = att_scores / (self.head_dim ** 0.5)
        
        # Apply causal mask: broadcast (T, T) -> (1, 1, T, T) across (B, H, T, T).
        # Positions where mask==True (future) get -inf so softmax -> 0.
        att_scores.masked_fill_(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), -torch.inf)
        
        # Convert to probabilities along the last dimension (over keys/time).
        att_weights = torch.softmax(att_scores, dim=-1)  # (B, H, T, T)

        # Regularize attention by dropping some probability mass.
        att_weights = self.dropout(att_weights)
        
        # Weighted sum of values: (B, H, T, T) @ (B, H, T, Hd) -> (B, H, T, Hd)
        context_vec = att_weights @ values

        # Move time back in front of heads: (B, H, T, Hd) -> (B, T, H, Hd)
        context_vec = context_vec.transpose(1, 2)

        # Merge heads: (B, T, H, Hd) -> (B, T, H*Hd=d_out)
        context_vec = context_vec.contiguous().view(B, T, self.d_out)
        
        # Final linear projection mixes head outputs.
        out = self.out_proj(context_vec)
        
        return out

class LayerNorm(nn.Module):
    """
    Layer Normalization over the last (feature) dimension.

    For each token (or element along the last dimension), LayerNorm normalizes
    by subtracting the mean and dividing by the standard deviation computed
    across that last dimension, then applies learnable scale (gamma) and
    shift (beta) parameters.

    Given input x with shape (..., D), we compute:
        mean = mean(x, dim=-1, keepdim=True)
        var  = var(x, dim=-1, keepdim=True)   # population variance
        x̂    = (x - mean) / sqrt(var + eps)
        y    = scale * x̂ + shift

    Args:
        emb_dim (int): Size of the last dimension D to be normalized.

    Attributes:
        eps (float): Small constant for numerical stability.
        scale (nn.Parameter): Learnable gain (gamma), shape (emb_dim,).
        shift (nn.Parameter): Learnable bias (beta), shape (emb_dim,).

    Shapes:
        Input:  x -> (B, T, emb_dim) or any shape with last dim = emb_dim
        Output: y -> same shape as input
    """

    def __init__(self, emb_dim: int):
        super().__init__()

        # Epsilon prevents division by zero when variance is very small.
        self.eps = 1e-5

        # Learnable affine parameters: gamma (scale) and beta (shift).
        # Initialized to gamma=1, beta=0 (identity transform).
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean over the last dimension (features), keep dims for broadcasting.
        mean = x.mean(dim=-1, keepdim=True)

        # Use *population* variance (unbiased=False). This matches standard LayerNorm.
        # keepdim=True so it broadcasts correctly when normalizing.
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize: (x - mean) / sqrt(var + eps). Numerical-stability via eps.
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable affine transform (broadcast over all leading dims).
        return self.scale * x_norm + self.shift
    
    
class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation.

    This implementation uses the widely adopted *tanh approximation*:
        GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

    It closely matches the exact GELU defined via the Gaussian CDF and is used in
    many Transformer architectures (e.g., BERT, GPT variants), offering a good
    trade-off between accuracy and speed.

    Shapes:
        Input:  x -> any shape
        Output: y -> same shape as x (elementwise activation)

    Notes:
        - If you prefer the exact GELU, use: 0.5 * x * (1 + erf(x / sqrt(2))).
        - PyTorch provides a built-in: nn.GELU(approximate='tanh') or F.gelu.
    """
    def __init__(self):
        super().__init__()  # Initialize base nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the tanh-based GELU approximation elementwise.
        # term = sqrt(2/pi) * (x + 0.044715 * x^3)
        term = torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype)) * (
            x + 0.044715 * torch.pow(x, 3)
        )
        # GELU ≈ 0.5 * x * (1 + tanh(term))
        return 0.5 * x * (1.0 + torch.tanh(term))
    
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used in Transformer blocks.

    This module applies two linear layers with a nonlinearity in between:
        FFN(x) = W2( GELU( W1(x) ) )
    with an expansion ratio (typically 4x) in the hidden layer.

    Args:
        cfg (dict): Configuration dictionary. Expected keys:
            - 'emb_dim' (int): The model/embedding dimension D.
            - 'ff_expansion' (int, optional): Expansion factor for hidden width
                  (default: 4, so hidden_dim = 4 * emb_dim).
            - 'dropout' (float, optional): Dropout probability applied after
                  the activation (default: 0.0).

    Shapes:
        Input:  x -> (B, T, D) or (N, D) where D == emb_dim
        Output: y -> same shape as x
    """
    def __init__(self, cfg):
        super().__init__()
        # Define the position-wise MLP:
        #   Linear(D -> hidden) -> GELU -> Dropout -> Linear(hidden -> D)
        # Sequential keeps it compact and readable.
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], cfg['emb_dim'] * 4),  # Expand feature dimension
            GELU(),                          # Nonlinear activation (tanh-approx GELU)           # Regularization on activations
            nn.Linear(cfg['emb_dim'] * 4, cfg['emb_dim']),  # Project back to emb_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input of shape (..., emb_dim)

        Returns:
            torch.Tensor: Same shape as input.
        """
        # Apply the two-layer MLP to each position independently.
        return self.layers(x)
    
# Standard Transformer block (Pre-LayerNorm) with residual connections.
# - Two LayerNorms (pre-norm style): one before attention, one before FFN.
# - Multi-head causal self-attention sublayer.
# - Position-wise feed-forward network (FFN) sublayer.
# - Residual (skip) connections around each sublayer.
# - Dropout after each sublayer for regularization.

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with causal multi-head attention.

    This block follows the common "pre-norm" design:
      1) LayerNorm → Multi-Head Self-Attention → Dropout → Residual add
      2) LayerNorm → Position-wise FeedForward → Dropout → Residual add

    The attention module is causal, so each position can only attend to
    current and past positions (autoregressive). Dropout is applied after
    each sublayer before the residual connection.

    Args:
        cfg (dict): Configuration dictionary with keys:
            - 'emb_dim' (int): Model/embedding dimension.
            - 'context_length' (int): Maximum sequence length.
            - 'n_heads' (int): Number of attention heads.
            - 'drop_rate' (float): Dropout probability.
            - 'qkv_bias' (bool): Whether to use bias in Q/K/V projections.

    Attributes:
        layernorm1 (LayerNorm): Pre-attention normalization.
        layernorm2 (LayerNorm): Pre-FFN normalization.
        attn (MultiHeadAttention): Causal multi-head self-attention.
        ffn (FeedForward): Position-wise MLP.
        dropout (nn.Dropout): Dropout after attention and FFN.
    """
    def __init__(self, cfg):
        super().__init__()

        # LayerNorm applied before attention (pre-norm). Normalizes the last dim (emb_dim).
        self.layernorm1 = LayerNorm(cfg['emb_dim'])

        # LayerNorm applied before the feed-forward network (pre-norm).
        self.layernorm2 = LayerNorm(cfg['emb_dim'])

        # Multi-head causal self-attention:
        #   d_in = d_out = model width (emb_dim)
        #   context_length = maximum sequence length for the causal mask
        #   num_heads = number of attention heads
        #   dropout = attention dropout (applied to attention weights)
        #   qkv_bias = whether to include bias terms in Q/K/V projections
        self.attn = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias = cfg['qkv_bias']
        )

        # Position-wise feed-forward network (typically expands dimension, applies GELU, projects back).
        self.ffn = FeedForward(cfg)

        # Dropout applied to sublayer outputs before adding residuals.
        self.dropout = nn.Dropout(cfg['drop_rate'])
        
    def forward(self, x):
        # ---- Sublayer 1: Attention + Residual ----
        shortcut = x                    # Save residual (input to attention path)
        x = self.layernorm1(x)          # Pre-norm: stabilize stats before attention
        x = self.attn(x)                # Causal multi-head self-attention over sequence
        x = self.dropout(x)             # Regularize attention output
        x = x + shortcut                # Residual connection (adds back original input)
        
        # ---- Sublayer 2: Feed-Forward + Residual ----
        shortcut = x                    # Save residual (input to FFN path)
        x = self.layernorm2(x)          # Pre-norm: stabilize stats before FFN
        x = self.ffn(x)                 # Position-wise MLP (applied independently at each position)
        x = self.dropout(x)             # Regularize FFN output
        x = x + shortcut                # Residual connection

        return x                        # Shape preserved: (B, T, emb_dim)
    
class GPTModel(nn.Module):
    """Decoder-only Transformer language model (GPT-style).

    This module embeds token IDs and absolute positions, sums them,
    applies a stack of Transformer blocks (causal self-attention + FFN),
    normalizes, and projects to vocabulary logits.

    Model layout (per forward pass):
        tokens --> token embeddings
               +  positional embeddings
               -> dropout
               -> N × TransformerBlock (pre-LN, causal MHA + FFN)
               -> LayerNorm
               -> linear head to vocab logits

    Args:
        cfg (dict): Configuration dictionary with required keys:
            - 'vocab_size' (int): Size of the tokenizer vocabulary.
            - 'emb_dim' (int): Embedding/model dimension (d_model).
            - 'context_length' (int): Maximum sequence length (for positions).
            - 'drop_rate' (float): Dropout probability.
            - 'n_layers' (int): Number of Transformer blocks.
            - 'n_heads' (int): Number of attention heads (consumed by blocks).
            - 'qkv_bias' (bool): Whether to use bias in Q/K/V projections; here
                                 reused for `lm_head` bias as provided.

    Attributes:
        tok_emb (nn.Embedding): Token embedding table (vocab_size × emb_dim).
        pos_emb (nn.Embedding): Positional embedding table
            (context_length × emb_dim).
        dropout (nn.Dropout): Dropout applied after embedding sum.
        transformer_blocks (nn.Sequential): Stack of TransformerBlock modules.
        layernorm (LayerNorm): Final LayerNorm before LM head.
        lm_head (nn.Linear): Output projection to vocabulary size.

    Shapes:
        Input:
            idx: LongTensor of shape (B, T) with token IDs in [0, vocab_size).
        Output:
            logits: FloatTensor of shape (B, T, vocab_size).

  """

    def __init__(self, cfg):
        super().__init__()

        # Embed tokens into continuous vectors of size emb_dim.
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])

        # Embed absolute positions 0..context_length-1 into emb_dim.
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])

        # Dropout on the sum of token+position embeddings (regularization).
        self.dropout = nn.Dropout(cfg['drop_rate'])

        # Stack of Transformer blocks (pre-LN, causal MHA + FFN).
        # nn.Sequential executes them in order.
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Final LayerNorm before projecting to logits (stabilizes activations).
        self.layernorm = LayerNorm(cfg['emb_dim'])

        # Linear projection to vocabulary logits for next-token prediction.
        # Bias usage follows cfg['qkv_bias'] as provided by the calling code.
        self.lm_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=cfg['qkv_bias'])
        
    def forward(self, idx):
        """Compute vocabulary logits for each input position.

        Args:
            idx (torch.LongTensor): Token IDs of shape (B, T).

        Returns:
            torch.FloatTensor: Logits of shape (B, T, vocab_size).
        """
        # Unpack batch (B) and sequence length (T) from input IDs.
        B, T = idx.shape

        # Token embeddings: (B, T, emb_dim)
        tok_emb = self.tok_emb(idx)

        # Positional embeddings for indices [0..T-1]: (T, emb_dim).
        # This will broadcast over batch (added to each sequence in the batch).
        pos_emd = self.pos_emb(torch.arange(T, device=idx.device))

        # Combine token and positional information: (B, T, emb_dim)
        x = tok_emb + pos_emd

        # Regularize with dropout at the embedding level.
        x = self.dropout(x)  # (B, T, emb_dim)

        # Apply stacked Transformer blocks with causal attention.
        x = self.transformer_blocks(x)  # (B, T, emb_dim)

        # Final normalization before output projection.
        x = self.layernorm(x)  # (B, T, emb_dim)

        # Project to vocabulary-sized logits per position.
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits
    
