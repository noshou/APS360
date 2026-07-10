import torch
import torch.nn.functional as F

from torch           import nn
from typing          import Callable
from jaxtyping       import jaxtyped
from beartype        import beartype
from beartype.typing import Tuple
from ..batching      import Batch
from Preprocess      import VOCAB
from .layer_head     import LayerHead
from ..utils.no_trilin_bilin import NoTrilinBilin

class Embed(nn.Module):
    
    """
    Embed each atom into a learned representation and estimate its scattering strength.

    Produces three outputs per atom:
        - embeds: learned identity embedding, shape (N, M, 1, λ₁)
        - f_mags: form factor magnitude |f(q)| per q-point, shape (N, M, Q, 1)
        - sigmas: RFF kernel bandwidth per q-point, shape (N, M, Q, 1)
    """

    _mbd:    nn.Embedding # atom identity in learned space
    _f0f1:   nn.Linear    # approximate real form factor at each q
    _f2:     nn.Linear    # approximate imaginary form factor at each q
    _prelu:  nn.PReLU     # prelu activation function
    _sigma:  NoTrilinBilin  # computes positional scaling factor based on embed and f_mag
    _fwd_fn: Callable     # torch.compiled or plain _forward_fn, per the compile flag

    def __init__(self, lambda_1: int, qPoints: int, compile: bool = False) -> None:
        
        """Construct the embedding, form-factor, and sigma-estimation layers.

        Parameters
        ----------
        lambda_1 : int
            Embedding dimension (λ₁).
        qPoints : int
            Number of q-points (Q).
        compile : bool, optional
            If True, torch.compile `_forward_fn`. Default is False.

        Returns
        -------
        None
        """
        
        super().__init__()
        self._mbd    = nn.Embedding(len(VOCAB)+1, lambda_1, padding_idx=0) # +1 since 0 is padding idx
        self._f0f1   = nn.Linear(lambda_1, qPoints)
        self._f2     = nn.Linear(lambda_1, qPoints)
        self._prelu  = nn.PReLU(lambda_1)
        # NoTrilinBilin, not nn.Bilinear: the F.bilinear op dispatches to aten::_trilinear,
        # which profiled as a top CUDA op here (741ms / 12 calls) and forces a torch.compile
        # graph break. NoTrilinBilin is mathematically identical (verified to fp32 rounding)
        # with the same init, routing through GEMM + elementwise instead. NOTE: this is the
        # (out=Q, in2=Q) "both moderate" case the NoTrilinBilin docstring flags - it
        # materializes a (N, M, Q, Q) temp nn.Bilinear never allocates, so confirm the net
        # win on the profiler rather than assuming it.
        self._sigma  = NoTrilinBilin(lambda_1, qPoints, qPoints)
        # mode="reduce-overhead": see the matching comment in MessagePass - launch-bound
        # profile (Self CPU >> Self CUDA), CUDA graphs cut per-kernel dispatch overhead.
        self._fwd_fn = torch.compile(self._forward_fn, dynamic=True, fullgraph=True, mode="reduce-overhead") if compile else self._forward_fn
    
    @staticmethod
    def _forward_fn(
        prelu:  nn.PReLU,
        mbd:    nn.Embedding,
        f0f1:   nn.Linear,
        f2:     nn.Linear,
        sigma:  NoTrilinBilin,
        vocabs: torch.Tensor,
        mask:   torch.Tensor,
        eps:    float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """Compute atom embeddings, form factor magnitudes, and sigmas.

        Parameters
        ----------
        prelu : torch.nn.PReLU
            Activation applied to the raw atom embedding.
        mbd : torch.nn.Embedding
            Atom identity embedding table.
        f0f1 : torch.nn.Linear
            Linear layer approximating the real part of the form factor.
        f2 : torch.nn.Linear
            Linear layer approximating the imaginary part of the form
            factor.
        sigma : NoTrilinBilin
            Bilinear layer (nn.Bilinear-equivalent) computing the positional
            scaling factor from the embedding and form factor magnitude.
        vocabs : torch.Tensor
            Atom vocabulary indices, shape (N, M).
        mask : torch.Tensor
            Padding mask, shape (N, M); True marks a real (non-padding)
            atom.
        eps : float
            Numerical floor added to form factor magnitudes and sigmas.

        Returns
        -------
        torch.Tensor
            `embeds`, atom identity representations, shape (N, M, 1, λ₁).
        torch.Tensor
            `f_mags`, form factor magnitudes, shape (N, M, Q, 1).
        torch.Tensor
            `sigmas`, kernel variance bandwidths, shape (N, M, Q, 1).
        """
        
        # get embedding: "what is this atom?" → (N, M, λ₁)
        # since linear layers apply to last dimension, must keep this order stable
        # since we want to activate using prelu, but nn.PReLU() acts on dim=1 for |dim| >= 2, 
        # first transpose embed to (N,λ₁,M), do prelu, then re-transpose to (N,M,λ₁)
        embed = (prelu(mbd(vocabs).transpose(-1, -2))).transpose(-1, -2)

        # estimate form factors → (N, M, Q)
        # as above, must keep this stable at 3 dimensions
        f_rel  = f0f1(embed) + eps
        f_img  = f2(embed)
        f_mag  = torch.hypot(f_rel, f_img) + eps

        # get padding mask (N,M) -> (N,M,1)
        masks = mask.unsqueeze(-1)

        # compute sigmas (N, M, Q, 1)
        sigmas = (F.softplus(sigma(embed, f_mag)) + eps).unsqueeze(-1) * masks.unsqueeze(-1)
        
        # squeeze padding mask (N,M,1) -> (N,M,1,1)
        masks = masks.unsqueeze(-1)
        
        # pad the layer heads to have correct dimensions, drop masked atoms:
        # embeds → (N, M, 1, λ₁); f_mags → (N, M, Q, 1)
        embeds = embed.unsqueeze(-2)
        f_mags = f_mag.unsqueeze(-1) * masks

        return embeds, f_mags, sigmas
    
    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, eps: float) -> LayerHead:
    
        """Embed a batch's atoms and estimate their scattering properties.

        Parameters
        ----------
        batch : Batch
            Input batch; uses `batch.vocab`, shape (N, M), atom indices.
        eps : float
            Numerical floor to avoid zero form factors and sigmas.

        Returns
        -------
        LayerHead
            Container with `embeds` (N, M, 1, λ₁), `f_mags` (N, M, Q, 1),
            and `sigmas` (N, M, Q, 1).
        """

        embeds, f_mags, sigmas = self._fwd_fn(
            self._prelu,
            self._mbd,
            self._f0f1,
            self._f2,
            self._sigma,
            batch.vocab, 
            batch.padding_mask(), 
            eps
        )
        
        return LayerHead(embeds=embeds, f_mags=f_mags, sigmas=sigmas)
