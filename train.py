import argparse
import h5py
import json
import torch

from dataclasses          import replace as dc_replace
from torch.utils.data     import DataLoader
from Preprocess           import Encoding
from ScatterNet           import ScatterNet
from ScatterNet.batching  import Batcher
from ScatterNet.config    import RunConfig, load_config
from ScatterNet.train     import Loss


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train ScatterNet. All flags override --config values."
    )
    p.add_argument("--config",        default=None,  help="path to YAML run config (e.g. runs/default.yaml)")
    # paths
    p.add_argument("--hdf5",          default=None)
    p.add_argument("--db",            default=None)
    p.add_argument("--ckpt_best",     default=None)
    p.add_argument("--ckpt_resume",   default=None)
    p.add_argument("--metrics",       default=None)
    p.add_argument("--resume",        default=None,  help="path to resume checkpoint")
    # model
    p.add_argument("--lambda_1",      type=int,   default=None)
    p.add_argument("--lambda_2",      type=int,   default=None)
    p.add_argument("--lambda_3",      type=int,   default=None)
    p.add_argument("--lambda_4",      type=int,   default=None)
    p.add_argument("--lambda_5",      type=int,   default=None)
    p.add_argument("--msg_seed",      type=int,   default=None)
    p.add_argument("--eps_embd",      type=float, default=None)
    p.add_argument("--eps_msgp",      type=float, default=None)
    # loss
    p.add_argument("--lambda_6",      type=float, default=None)
    p.add_argument("--lambda_7",      type=float, default=None)
    p.add_argument("--eps_sigma",     type=float, default=None)
    # training
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--weight_decay",  type=float, default=None)
    p.add_argument("--grad_clip",     type=float, default=None)
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batcher_seed",  type=int,   default=None)
    p.add_argument("--atom_size_ceil",type=int,   default=None)
    p.add_argument("--num_workers",   type=int,   default=None)
    return p.parse_args()


# ── eval helper ──────────────────────────────────────────────────────────────

def evaluate(loader, model, criterion, cfg: RunConfig, device: str):
    model.eval()
    total_loss = 0.0
    total_mols = 0
    ss_res = 0.0
    sum_y  = 0.0
    sum_y2 = 0.0
    n_elem = 0
    with torch.no_grad():
        for batch in loader:
            batch            = dc_replace(batch, vocab=batch.vocab.to(device), iqval=batch.iqval.to(device), coord=batch.coord.to(device))
            iq, fmags, sigmas = model(batch)
            loss             = criterion.loss(iq, fmags, sigmas, batch, cfg.lambda_6, cfg.lambda_7, cfg.eps_sigma)
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


# ── main ─────────────────────────────────────────────────────────────────────

def main(cfg: RunConfig | None = None):
    """Entry point. Pass a RunConfig directly (e.g. from a Kaggle notebook),
    or leave None to parse sys.argv."""

    if cfg is None:
        A = _parse_args()
        cfg = load_config(
            A.config,
            hdf5          = A.hdf5,
            db            = A.db,
            ckpt_best     = A.ckpt_best,
            ckpt_resume   = A.ckpt_resume,
            metrics       = A.metrics,
            resume        = A.resume,
            lambda_1      = A.lambda_1,
            lambda_2      = A.lambda_2,
            lambda_3      = A.lambda_3,
            lambda_4      = A.lambda_4,
            lambda_5      = A.lambda_5,
            msg_seed      = A.msg_seed,
            eps_embd      = A.eps_embd,
            eps_msgp      = A.eps_msgp,
            lambda_6      = A.lambda_6,
            lambda_7      = A.lambda_7,
            eps_sigma     = A.eps_sigma,
            lr            = A.lr,
            weight_decay  = A.weight_decay,
            grad_clip     = A.grad_clip,
            epochs        = A.epochs,
            batcher_seed  = A.batcher_seed,
            atom_size_ceil= A.atom_size_ceil,
            num_workers   = A.num_workers,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"config: lr={cfg.lr}  epochs={cfg.epochs}  λ1={cfg.lambda_1}  λ2={cfg.lambda_2}  λ3={cfg.lambda_3}  λ4={cfg.lambda_4}  λ5={cfg.lambda_5}  λ6={cfg.lambda_6}  λ7={cfg.lambda_7}  eps_σ={cfg.eps_sigma}")

    # ── data ──────────────────────────────────────────────────────────────────

    with h5py.File(cfg.hdf5, "r") as f:
        q_grid = torch.tensor(f["q_grid"][:]).float()   # type: ignore[index]
    energy   = 12_500.0  # eV; fixed pipeline constant
    q_points = len(q_grid)

    enc = Encoding(cfg.db, cfg.hdf5)

    batcher = Batcher(
        hdf5_db        = cfg.hdf5,
        enc            = enc,
        batches        = cfg.buckets,
        seed           = cfg.batcher_seed,
        atom_size_ceil = cfg.atom_size_ceil,
    )
    train_set, val_set, test_set = batcher.get_sets()

    pin   = device == "cuda"
    pw    = cfg.num_workers > 0
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  collate_fn=lambda x: x[0], num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=pw)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=pw)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=pw)

    # ── model ─────────────────────────────────────────────────────────────────

    model = ScatterNet(
        lambda_1 = cfg.lambda_1,
        lambda_2 = cfg.lambda_2,
        lambda_3 = cfg.lambda_3,
        lambda_4 = cfg.lambda_4,
        lambda_5 = cfg.lambda_5,
        msg_seed = cfg.msg_seed,
        q_points = q_points,
        eps_embd = cfg.eps_embd,
        eps_msgp = cfg.eps_msgp,
    ).to(device)

    criterion = Loss(q_grid, energy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 1
    history: list = []
    best_val = float("inf")

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("val_loss", float("inf"))
        print(f"resumed from {cfg.resume} (epoch {ckpt['epoch']}, val_loss {best_val:.4f})")

    # ── training loop ─────────────────────────────────────────────────────────

    hparams = dict(
        lambda_1=cfg.lambda_1, lambda_2=cfg.lambda_2,
        lambda_3=cfg.lambda_3, lambda_4=cfg.lambda_4,
        lambda_5=cfg.lambda_5, msg_seed=cfg.msg_seed,
        q_points=q_points,     eps_embd=cfg.eps_embd,
        eps_msgp=cfg.eps_msgp,
    )

    for epoch in range(start_epoch, start_epoch + cfg.epochs):

        model.train()
        train_loss_sum = 0.0
        train_mols     = 0
        train_ss_res   = 0.0
        train_sum_y    = 0.0
        train_sum_y2   = 0.0
        train_n_elem   = 0

        for _bi, batch in enumerate(train_loader):
            if cfg.max_batches is not None and _bi >= cfg.max_batches:
                break
            batch             = dc_replace(batch, vocab=batch.vocab.to(device), iqval=batch.iqval.to(device), coord=batch.coord.to(device))
            iq, fmags, sigmas = model(batch)
            loss              = criterion.loss(iq, fmags, sigmas, batch, cfg.lambda_6, cfg.lambda_7, cfg.eps_sigma)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            n = batch.iqval.shape[0]
            train_loss_sum += loss.item() * n
            train_mols     += n
            with torch.no_grad():
                log_pred      = torch.log1p(iq)
                log_target    = torch.log1p(batch.iqval)
                train_ss_res += ((log_pred - log_target) ** 2).sum().item()
                train_sum_y  += log_target.sum().item()
                train_sum_y2 += (log_target ** 2).sum().item()
                train_n_elem += log_target.numel()

        train_loss   = train_loss_sum / train_mols
        train_ss_tot = train_sum_y2 - train_sum_y ** 2 / train_n_elem
        train_r2     = 1.0 - train_ss_res / train_ss_tot if train_ss_tot > 0 else 0.0

        val_loss,  val_r2  = evaluate(val_loader,  model, criterion, cfg, device)
        test_loss, test_r2 = evaluate(test_loader, model, criterion, cfg, device)

        print(
            f"epoch {epoch:3d}"
            f"  train loss {train_loss:.4f}  r2 {train_r2:.4f}"
            f"  |  val loss {val_loss:.4f}  r2 {val_r2:.4f}"
            f"  |  test loss {test_loss:.4f}  r2 {test_r2:.4f}"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_r2":   train_r2,
            "val_loss":   val_loss,
            "val_r2":     val_r2,
            "test_loss":  test_loss,
            "test_r2":    test_r2,
        })
        with open(cfg.metrics, "w") as fh:
            json.dump({"epochs": history}, fh, indent=2)

        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss":  val_loss,
            "q_grid":    q_grid,
            "energy":    energy,
            "hparams":   hparams,
        }, cfg.ckpt_resume)

        if val_loss < best_val:
            best_val = val_loss
            fp16_state = {k: v.half() for k, v in model.state_dict().items()}
            torch.save({
                "epoch":    epoch,
                "model":    fp16_state,
                "val_loss": val_loss,
                "q_grid":   q_grid,
                "energy":   energy,
                "hparams":  hparams,
            }, cfg.ckpt_best)
            print(f"  saved best checkpoint (val {val_loss:.4f})")

    with open(cfg.metrics, "w") as fh:
        json.dump({"epochs": history}, fh, indent=2)


if __name__ == "__main__":
    main()
