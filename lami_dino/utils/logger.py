# Copyright (c) 2025 Zhengjian Kang
# Research logger for LaMI-DETR family (TPA + APR + RPSA)

import csv
import torch
import torch.nn.functional as F
from pathlib import Path

class TrainLogger:
    """
    Periodically record dynamic training indicators:
        - loss_apr, loss_orth, loss_div
        - prototype orthogonality
        - prototype usage entropy
        - (optional) base/novel mAP if available
    Output: CSV file under cfg.train.output_dir / training_log.csv
    """

    def __init__(self, save_dir, interval=200):
        self.interval = interval
        self.save_path = Path(save_dir) / "training_log.csv"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._iter = 0

    def log(self, iteration, model, losses):
        if self._iter % self.interval != 0:
            self._iter += 1
            return
        self._iter += 1

        record = {"iter": iteration}
        # ==== 1. record losses ====
        for k, v in losses.items():
            try:
                record[k] = float(v)
            except Exception:
                continue

        # ==== 2. prototype metrics ====
        try:
            tpa = model.transformer.decoder.class_embed[0].tpa
            if hasattr(tpa, "last_prototypes") and hasattr(tpa, "last_attention"):
                P = F.normalize(tpa.last_prototypes, dim=-1)
                gram = torch.einsum("ckd,cmd->ckm", P, P)
                off_diag = (gram - torch.eye(gram.size(-1), device=gram.device)).abs().mean()
                usage = tpa.last_attention.mean(dim=-1).mean()
                record["orthogonality"] = float(off_diag)
                record["usage_entropy"] = float(usage)
        except Exception:
            pass

        # ==== 3. write CSV ====
        write_header = not self._initialized or not self.save_path.exists()
        with self.save_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
                self._initialized = True
            writer.writerow(record)

        if iteration % (self.interval * 5) == 0:
            print(f"[TrainLogger] Iter {iteration}: saved metrics -> {self.save_path}")



# import pandas as pd, matplotlib.pyplot as plt
# df = pd.read_csv("training_log.csv")
# plt.plot(df["iter"], df["loss_apr"], label="loss_apr")
# plt.plot(df["iter"], df["orthogonality"], label="orthogonality")
# plt.plot(df["iter"], df["usage_entropy"], label="usage_entropy")
# plt.legend(); plt.xlabel("Iter"); plt.ylabel("Value"); plt.show()
