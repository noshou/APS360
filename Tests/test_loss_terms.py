"""Unit tests for Loss._kratky_MSLE and Loss._ff_penalty."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from ScatterNet.utils   import Loss
from ScatterNet.batching import Batch

Q = 6

def make_loss():
    return Loss(torch.linspace(0.1, 1.0, Q), 12500.0)

def make_batch(vocab, iqval, coord):
    return Batch.from_lists(
        [torch.tensor(v) for v in vocab],
        [torch.tensor(i, dtype=torch.float32) for i in iqval],
        [torch.tensor(c, dtype=torch.float32) for c in coord],
    )

# ── _kratky_MSLE ──────────────────────────────────────────────────────────────

def test_kratky_perfect_prediction_is_zero():
    """Predicting exact ground truth must yield zero loss everywhere."""
    iq_gt = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    batch = make_batch(
        vocab=[[1]], iqval=[iq_gt], coord=[[[0., 0., 0.]]],
    )
    loss = make_loss()
    pred = batch.iqval.clone()
    out  = loss._kratky_MSLE(pred, batch)
    assert out.shape == (1, Q)
    assert torch.allclose(out, torch.zeros(1, Q), atol=1e-6), f"non-zero at perfect pred: {out}"
    print(f"  max_err={out.abs().max().item():.2e}  ✓")

def test_kratky_high_q_gets_more_weight():
    """Kratky weight (1+q²) must be strictly increasing in q."""
    loss = make_loss()
    weights = loss._q_weights_.squeeze()  # (Q,)
    diffs = weights[1:] - weights[:-1]
    assert (diffs > 0).all(), f"weights not monotone: {weights}"
    print(f"  weights={weights.tolist()}  ✓")

def test_kratky_increases_with_error():
    """Larger prediction error must give strictly larger loss."""
    gt    = [1.0] * Q
    batch = make_batch([[1]], [gt], [[[0.,0.,0.]]])
    loss  = make_loss()

    pred_close = torch.tensor([[v * 1.1 for v in gt]])
    pred_far   = torch.tensor([[v * 5.0 for v in gt]])

    out_close = loss._kratky_MSLE(pred_close, batch).mean().item()
    out_far   = loss._kratky_MSLE(pred_far,   batch).mean().item()
    assert out_far > out_close, f"expected far ({out_far:.4f}) > close ({out_close:.4f})"
    print(f"  close={out_close:.4f}  far={out_far:.4f}  ✓")

def test_kratky_nonnegative():
    """Loss must be ≥ 0 for any prediction."""
    batch = make_batch([[1, 2]], [[0.1]*Q], [[[0.,0.,0.],[1.,0.,0.]]])
    loss  = make_loss()
    pred  = torch.rand(1, Q) * 10
    out   = loss._kratky_MSLE(pred, batch)
    assert (out >= 0).all(), f"negative entries: {out}"
    print(f"  min={out.min().item():.4f}  ✓")


# ── _ff_penalty ───────────────────────────────────────────────────────────────

def test_ff_penalty_zero_when_exact():
    """Predicting the tabulated form factors must give zero penalty."""
    loss  = make_loss()
    batch = make_batch([[1]], [[0.1]*Q], [[[0.,0.,0.]]])
    # ground truth form factors for vocab index 1
    f_real = loss._fmag_table[batch.vocab]  # (1, 1, Q)
    out = loss._ff_penalty(f_real, batch, lambda_6=1.0)
    assert out.shape == (1, Q)
    assert torch.allclose(out, torch.zeros(1, Q), atol=1e-5), f"non-zero at exact: {out}"
    print(f"  max_err={out.abs().max().item():.2e}  ✓")

def test_ff_penalty_scales_with_lambda6():
    """Doubling lambda_6 must double the penalty."""
    loss  = make_loss()
    batch = make_batch([[1]], [[0.1]*Q], [[[0.,0.,0.]]])
    pred  = torch.ones(1, 1, Q) * 5.0   # some wrong prediction

    p1 = loss._ff_penalty(pred, batch, lambda_6=1.0).mean().item()
    p2 = loss._ff_penalty(pred, batch, lambda_6=2.0).mean().item()
    assert abs(p2 / p1 - 2.0) < 1e-5, f"expected 2× got {p2/p1:.6f}"
    print(f"  λ6=1→{p1:.4f}  λ6=2→{p2:.4f}  ratio={p2/p1:.4f}  ✓")

def test_ff_padding_excluded():
    """Padded atoms must not contribute to the FF penalty."""
    loss = make_loss()
    # mol A: 1 atom;  mol B: 2 atoms — but same single real atom (idx 1)
    batch = make_batch(
        vocab=[[1], [1, 2]],
        iqval=[[0.1]*Q, [0.1]*Q],
        coord=[[[0.,0.,0.]], [[0.,0.,0.],[1.,0.,0.]]],
    )
    # set f_mag_pred equal to the real table values for ALL slots (including padding)
    f_real = loss._fmag_table[batch.vocab]  # (2, 2, Q)
    out = loss._ff_penalty(f_real, batch, lambda_6=1.0)
    # both rows should be zero — padding atom contributes nothing
    assert torch.allclose(out, torch.zeros(2, Q), atol=1e-5), \
        f"padding leaked: {out}"
    print(f"  max_err={out.abs().max().item():.2e}  ✓")

def test_ff_penalty_normalized_by_natoms():
    """Per-atom normalization: doubling atoms with same per-atom error → same penalty."""
    loss  = make_loss()
    wrong = torch.ones(1, 1, Q) * 5.0

    batch1 = make_batch([[1]],    [[0.1]*Q],    [[[0.,0.,0.]]])
    batch2 = make_batch([[1, 1]], [[0.1]*Q],    [[[0.,0.,0.],[1.,0.,0.]]])
    wrong2 = wrong.expand(1, 2, Q).clone()

    p1 = loss._ff_penalty(wrong,  batch1, lambda_6=1.0).mean().item()
    p2 = loss._ff_penalty(wrong2, batch2, lambda_6=1.0).mean().item()
    assert abs(p1 - p2) < 1e-5, f"normalization failed: 1-atom={p1:.4f} 2-atom={p2:.4f}"
    print(f"  1-atom={p1:.4f}  2-atom={p2:.4f}  ✓")


if __name__ == "__main__":
    tests = [
        ("kratky: perfect prediction → 0",      test_kratky_perfect_prediction_is_zero),
        ("kratky: high-q gets more weight",      test_kratky_high_q_gets_more_weight),
        ("kratky: loss increases with error",     test_kratky_increases_with_error),
        ("kratky: always non-negative",          test_kratky_nonnegative),
        ("ff: exact prediction → 0",             test_ff_penalty_zero_when_exact),
        ("ff: scales with λ6",                   test_ff_penalty_scales_with_lambda6),
        ("ff: padding excluded",                 test_ff_padding_excluded),
        ("ff: normalized by n_atoms",            test_ff_penalty_normalized_by_natoms),
    ]
    for name, fn in tests:
        print(name)
        fn()
    print("\nall tests passed")
