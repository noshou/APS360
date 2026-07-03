"""Unit tests for Loss._sg_penalty."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ScatterNet.utils import Loss
from ScatterNet.batching import Batch

def make_batch(vocab, iqval, coord):
    """Build a padded Batch from per-molecule lists of vocab, I(q), and coordinates.

    Parameters
    ----------
    vocab : list of list of int
        Per-molecule atom vocabulary indices.
    iqval : list of list of float
        Per-molecule ground-truth I(q) values.
    coord : list of list of list of float
        Per-molecule atom coordinates, shape (n_atoms, 3) per molecule.

    Returns
    -------
    Batch
        Batched, padded representation built via `Batch.from_lists`.
    """
    return Batch.from_lists(
        [torch.tensor(v) for v in vocab],
        [torch.tensor(i, dtype=torch.float32) for i in iqval],
        [torch.tensor(c, dtype=torch.float32) for c in coord],
    )

Q = 4  # tiny q-grid for all tests

def test_penalty_decreases_as_sigma_grows():
    """Larger sigma → smaller inverse-L1 penalty."""
    batch = make_batch(
        vocab = [[1], [1]],
        iqval = [[0.1]*Q, [0.1]*Q],
        coord = [[[0.,0.,0.]], [[0.,0.,0.]]],
    )
    loss = Loss(torch.linspace(0.1, 1.0, Q), 12500.0)

    sigmas_small = torch.full((2, 1, Q), 0.01)
    sigmas_large = torch.full((2, 1, Q), 1.00)

    pen_small = loss._sg_penalty(sigmas_small, lambda_7=1.0, batch=batch).mean().item()
    pen_large = loss._sg_penalty(sigmas_large, lambda_7=1.0, batch=batch).mean().item()

    assert pen_small > pen_large, f"expected pen_small ({pen_small:.4f}) > pen_large ({pen_large:.4f})"
    print(f"  penalty(σ=0.01)={pen_small:.4f}  penalty(σ=1.0)={pen_large:.4f}  ✓")

def test_padding_atoms_excluded():
    """Padded atoms (vocab=0) must not contribute to the penalty."""
    # mol A: 1 real atom;  mol B: 2 real atoms - same sigma
    batch = make_batch(
        vocab = [[1], [1, 2]],
        iqval = [[0.1]*Q, [0.1]*Q],
        coord = [[[0.,0.,0.]], [[0.,0.,0.],[1.,0.,0.]]],
    )
    loss = Loss(torch.linspace(0.1, 1.0, Q), 12500.0)

    # after Batch.from_lists, M=2; mol A has one padding atom (vocab=0, mask=0)
    sigmas = torch.full((2, 2, Q), 1.0)

    pen = loss._sg_penalty(sigmas, lambda_7=1.0, batch=batch)
    # both molecules should give the same per-atom average (lambda_7 / sigma = 1.0)
    assert torch.allclose(pen[0], pen[1], atol=1e-5), \
        f"mol A penalty {pen[0]} ≠ mol B penalty {pen[1]} - padding leaked in"
    print(f"  mol_A={pen[0].mean().item():.4f}  mol_B={pen[1].mean().item():.4f}  ✓")

def test_lambda7_scales_linearly():
    """Doubling lambda_7 should double the penalty."""
    batch = make_batch(
        vocab = [[1, 2]],
        iqval = [[0.1]*Q],
        coord = [[[0.,0.,0.],[1.,0.,0.]]],
    )
    loss = Loss(torch.linspace(0.1, 1.0, Q), 12500.0)
    sigmas = torch.full((1, 2, Q), 0.5)

    pen1 = loss._sg_penalty(sigmas, lambda_7=1.0, batch=batch).mean().item()
    pen2 = loss._sg_penalty(sigmas, lambda_7=2.0, batch=batch).mean().item()

    assert abs(pen2 / pen1 - 2.0) < 1e-5, f"expected 2x scaling, got {pen2/pen1:.6f}"
    print(f"  λ7=1→{pen1:.4f}  λ7=2→{pen2:.4f}  ratio={pen2/pen1:.4f}  ✓")

if __name__ == "__main__":
    print("test_penalty_decreases_as_sigma_grows")
    test_penalty_decreases_as_sigma_grows()
    print("test_padding_atoms_excluded")
    test_padding_atoms_excluded()
    print("test_lambda7_scales_linearly")
    test_lambda7_scales_linearly()
    print("all tests passed")
