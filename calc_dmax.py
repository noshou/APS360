"""
Compute D_max (maximum pairwise atomic distance) for every XYZ file.

Uses a 2-pass farthest-point algorithm (O(N)):
  1. Pick atom 0, find the farthest atom A from it.
  2. Find the farthest atom B from A.
  3. D_max = dist(A, B).
Exact for convex point sets; tight lower bound otherwise.

Output: CSV with columns [xyz_path, group, n_atoms, d_max]
"""

import os
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from tqdm import tqdm
import sys
from dataloads import groups

OUT     = "/mnt/c/home/nathan/APS360/data/dmax.csv"
WORKERS = max(1, (os.cpu_count() or 4) - 2)


def _dmax(coords: np.ndarray) -> float:
    def farthest_from(idx: int) -> tuple[int, float]:
        dists = np.sqrt(((coords - coords[idx]) ** 2).sum(axis=1))
        far = int(dists.argmax())
        return far, float(dists[far])

    a, _ = farthest_from(0)
    _, d = farthest_from(a)
    return d


def _process(xyz_path: str) -> tuple[str, int, float] | None:
    try:
        with open(xyz_path) as f:
            lines = f.readlines()
        # parse atom count from first two lines (handle both XYZ variants)
        try:
            n = int(lines[0])
            start = 2
        except ValueError:
            n = int(lines[1])
            start = 2
        if n == 0:
            return None
        xs, ys, zs = [], [], []
        for line in lines[start:start + n]:
            parts = line.split()
            if len(parts) < 4:
                return None
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
            zs.append(float(parts[3]))
        coords = np.array([xs, ys, zs], dtype=np.float32).T  # (N, 3)
        if len(coords) != n:
            return None
        d_max = _dmax(coords)
        return xyz_path, n, d_max
    except Exception:
        return None


if __name__ == "__main__":
    all_xyz = [
        (gname, path)
        for gname, gdir in groups.items()
        for path in (e.path for e in os.scandir(gdir) if e.name.endswith(".xyz"))
    ]
    total = len(all_xyz)

    with open(OUT, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["xyz_path", "group", "n_atoms", "d_max"])

        _inflight = WORKERS * 4
        it = iter(all_xyz)

        with ProcessPoolExecutor(max_workers=WORKERS) as pool, tqdm(
            total=total, unit="mol", file=sys.stdout, dynamic_ncols=True
        ) as pbar:
            futures = {
                pool.submit(_process, path): (gname, path)
                for gname, path in islice(it, _inflight)
            }

            while futures:
                fut = next(as_completed(futures))
                gname, path = futures.pop(fut)
                nxt = next(it, None)
                if nxt is not None:
                    ng, np_ = nxt
                    futures[pool.submit(_process, np_)] = (ng, np_)
                pbar.update()
                result = fut.result()
                if result is None:
                    continue
                xyz_path, n_atoms, d_max = result
                writer.writerow([xyz_path, gname, n_atoms, f"{d_max:.3f}"])
