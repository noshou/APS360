import h5py
import json
import torch

from dataclasses          import replace as dc_replace
from torch.utils.data     import DataLoader
from Preprocess           import Encoding
from ScatterNet           import ScatterNet
from ScatterNet.batching  import Batcher
from ScatterNet.train     import Loss

# ── hyperparameters ──────────────────────────────────────────────────────────

HDF5_PATH  = "I(q)@L=50.h5"
DB_NAME    = "scatternet"
CKPT_BEST   = "scatternet_best.pt"   # lean inference checkpoint (fp16 weights only)
CKPT_RESUME = "scatternet_resume.pt" # full state for resuming training (overwritten each epoch)
METRICS_PATH = "scatternet_metrics.json"

# model
LAMBDA_1   = 64       # atom embedding dim
LAMBDA_2   = 3        # message passing rounds
LAMBDA_3   = 128      # OutputHead hidden width (must be power-of-2-friendly with LAMBDA_4)
LAMBDA_4   = 3        # halving steps in OutputHead MLP (lambda_3 / 2^lambda_4 must reach 1)
LAMBDA_5   = 256      # RFF feature count
MSG_SEED   = 42       # seed for frozen RFF frequencies
EPS_EMBD   = 1e-8
EPS_MSGP   = 1e-8

# loss
LAMBDA_6   = 0.1      # form factor penalty weight

# training
LR         = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP  = 1.0
EPOCHS     = 50
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# atom-count buckets: list of (min_atoms, max_atoms)
# tune to match your dataset distribution and GPU memory
BUCKETS    = []

BATCHER_SEED      = 0
ATOM_SIZE_CEIL    = -1   # -1 = auto (3x max atom count in bucket)

# ── setup ────────────────────────────────────────────────────────────────────

with h5py.File(HDF5_PATH, "r") as f:
    q_grid = torch.tensor(f["q_grid"][:]).float()       # type: ignore[index]
    energy = float(f["energy"][()])                     # type: ignore[index]

q_points = len(q_grid)

enc = Encoding(DB_NAME, HDF5_PATH)

batcher = Batcher(
    hdf5_db        = HDF5_PATH,
    enc            = enc,
    batches        = BUCKETS,
    seed           = BATCHER_SEED,
    atom_size_ceil = ATOM_SIZE_CEIL,
)
train_set, val_set, test_set = batcher.get_sets()

train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  collate_fn=lambda x: x[0])
val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

model = ScatterNet(
    lambda_1 = LAMBDA_1,
    lambda_2 = LAMBDA_2,
    lambda_3 = LAMBDA_3,
    lambda_4 = LAMBDA_4,
    lambda_5 = LAMBDA_5,
    msg_seed = MSG_SEED,
    q_points = q_points,
    eps_embd = EPS_EMBD,
    eps_msgp = EPS_MSGP,
).to(DEVICE)

criterion = Loss(q_grid, energy).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ── metrics helpers ──────────────────────────────────────────────────────────

def evaluate(loader, model, criterion):
    """Return (loss, r2) over a DataLoader. R2 computed on log1p(I(q))."""
    model.eval()
    total_loss = 0.0
    total_mols = 0
    ss_res = 0.0
    sum_y  = 0.0
    sum_y2 = 0.0
    n_elem = 0
    with torch.no_grad():
        for batch in loader:
            batch     = dc_replace(batch, vocab=batch.vocab.to(DEVICE), iqval=batch.iqval.to(DEVICE), coord=batch.coord.to(DEVICE))
            iq, fmags = model(batch)
            loss      = criterion.loss(iq, fmags, batch, LAMBDA_6)
            n = batch.iqval.shape[0]
            total_loss += loss.item() * n
            total_mols += n
            log_pred   = torch.log1p(iq)
            log_target = torch.log1p(batch.iqval)
            ss_res += ((log_pred - log_target) ** 2).sum().item()
            sum_y  += log_target.sum().item()
            sum_y2 += (log_target ** 2).sum().item()
            n_elem += log_target.numel()
    mean_loss = total_loss / total_mols
    ss_tot    = sum_y2 - sum_y ** 2 / n_elem
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mean_loss, r2

# ── training loop ────────────────────────────────────────────────────────────

best_val = float("inf")
history  = []

for epoch in range(1, EPOCHS + 1):

    # train — accumulate loss and R2 stats in one pass
    model.train()
    train_loss_sum = 0.0
    train_mols     = 0
    train_ss_res   = 0.0
    train_sum_y    = 0.0
    train_sum_y2   = 0.0
    train_n_elem   = 0
    for batch in train_loader:
        batch     = dc_replace(batch, vocab=batch.vocab.to(DEVICE), iqval=batch.iqval.to(DEVICE), coord=batch.coord.to(DEVICE))
        iq, fmags = model(batch)
        loss      = criterion.loss(iq, fmags, batch, LAMBDA_6)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        n = batch.iqval.shape[0]
        train_loss_sum += loss.item() * n
        train_mols     += n
        with torch.no_grad():
            log_pred          = torch.log1p(iq)
            log_target        = torch.log1p(batch.iqval)
            train_ss_res     += ((log_pred - log_target) ** 2).sum().item()
            train_sum_y      += log_target.sum().item()
            train_sum_y2     += (log_target ** 2).sum().item()
            train_n_elem     += log_target.numel()

    train_loss   = train_loss_sum / train_mols
    train_ss_tot = train_sum_y2 - train_sum_y ** 2 / train_n_elem
    train_r2     = 1.0 - train_ss_res / train_ss_tot if train_ss_tot > 0 else 0.0

    val_loss, val_r2 = evaluate(val_loader, model, criterion)

    print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  r2 {train_r2:.4f}  |  val loss {val_loss:.4f}  r2 {val_r2:.4f}")

    history.append({
        "epoch":      epoch,
        "train_loss": train_loss,
        "train_r2":   train_r2,
        "val_loss":   val_loss,
        "val_r2":     val_r2,
    })
    with open(METRICS_PATH, "w") as f:
        json.dump({"epochs": history}, f, indent=2)

    # always overwrite resume checkpoint so training can be continued from last epoch
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
        "q_grid":     q_grid,
        "energy":     energy,
        "hparams": {
            "lambda_1": LAMBDA_1, "lambda_2": LAMBDA_2,
            "lambda_3": LAMBDA_3, "lambda_4": LAMBDA_4,
            "lambda_5": LAMBDA_5, "msg_seed": MSG_SEED,
            "q_points": q_points, "eps_embd": EPS_EMBD,
            "eps_msgp": EPS_MSGP,
        },
    }, CKPT_RESUME)

    if val_loss < best_val:
        best_val = val_loss
        fp16_state = {k: v.half() for k, v in model.state_dict().items()}
        torch.save({
            "epoch":    epoch,
            "model":    fp16_state,
            "val_loss": val_loss,
            "q_grid":   q_grid,
            "energy":   energy,
            "hparams": {
                "lambda_1": LAMBDA_1, "lambda_2": LAMBDA_2,
                "lambda_3": LAMBDA_3, "lambda_4": LAMBDA_4,
                "lambda_5": LAMBDA_5, "msg_seed": MSG_SEED,
                "q_points": q_points, "eps_embd": EPS_EMBD,
                "eps_msgp": EPS_MSGP,
            },
        }, CKPT_BEST)
        print(f"  saved best checkpoint (val {val_loss:.4f})")

# ── test (once, after training) ───────────────────────────────────────────────

test_loss, test_r2 = evaluate(test_loader, model, criterion)
print(f"test  loss {test_loss:.4f}  r2 {test_r2:.4f}")

with open(METRICS_PATH, "w") as f:
    json.dump({"epochs": history, "test": {"loss": test_loss, "r2": test_r2}}, f, indent=2)
