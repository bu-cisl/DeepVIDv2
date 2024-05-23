import csv
import os
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class CallbackSaveImage(object):
    def __init__(
        self, dataset, model, device, output_dir, num_sample=4, vmin=-2, vmax=6
    ):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.output_dir = output_dir
        if not os.path.isdir(os.path.join(self.output_dir, "preview")):
            os.mkdir(os.path.join(self.output_dir, "preview"))

        self.num_sample = num_sample

        sel_idx = np.array(
            len(self.dataset) / self.num_sample * np.arange(self.num_sample), dtype=int
        )

        self.X_sample = []
        self.y_true = []
        for i in sel_idx:
            X, y = self.dataset[i]
            y = y[0].detach().numpy()
            y = (y - np.nanmean(y)) / np.nanstd(y)

            self.X_sample.append(X)
            self.y_true.append(y)

        self.X_sample = torch.stack(self.X_sample, dim=0).to(self.device)
        self.y_true = np.concatenate(self.y_true, axis=1)

        self.vmin = vmin
        self.vmax = vmax

    def save(self, batch_id):
        self.model.eval()
        y_pred = self.model(self.X_sample)
        y_pred = y_pred.cpu().detach().numpy()

        y_pred_plot = []
        for y in y_pred:
            y = y[0]
            y = (y - np.nanmean(y)) / np.nanstd(y)
            y_pred_plot.append(y)
        y_pred_plot = np.concatenate(y_pred_plot, axis=1)

        img = np.concatenate([y_pred_plot, self.y_true], axis=0).squeeze()
        plt.imsave(
            os.path.join(self.output_dir, "preview", "batch_{:d}.png".format(batch_id)),
            img,
            vmin=self.vmin,
            vmax=self.vmax,
        )


class CallbackSaveModel(object):
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        if not os.path.isdir(os.path.join(self.output_dir, "checkpoint")):
            os.mkdir(os.path.join(self.output_dir, "checkpoint"))

        self.state = {
            "model": None,
            "optimizer": None,
        }

    def update(self, model, optimizer):
        self.state["model"] = model.state_dict()
        self.state["optimizer"] = optimizer.state_dict()

    def save(self, batch_idx: Optional[Union[int, str]] = None) -> None:
        if batch_idx is None:
            torch.save(
                self.state,
                os.path.join(self.output_dir, "checkpoint", "checkpoint_final.ckpt"),
            )
        elif type(batch_idx) is str:
            torch.save(
                self.state,
                os.path.join(
                    self.output_dir,
                    "checkpoint",
                    "checkpoint_{}.ckpt".format(batch_idx),
                ),
            )
        elif type(batch_idx) is int:
            torch.save(
                self.state,
                os.path.join(
                    self.output_dir,
                    "checkpoint",
                    "checkpoint_batch_{:d}.ckpt".format(batch_idx),
                ),
            )
        else:
            raise TypeError()


class Logger(object):
    def __init__(self, output_dir: str, use_tensorboard: bool = True) -> None:
        self.output_dir = output_dir
        if not os.path.isdir(os.path.join(self.output_dir)):
            os.mkdir(os.path.join(self.output_dir))

        self.metrics = []

        if use_tensorboard is True:
            self.tensorboard_writer = SummaryWriter(
                log_dir=os.path.join(output_dir, "tensorboard")
            )
        else:
            self.tensorboard_writer = None

    def update(
        self, metrics_dict: Dict[str, float], step: Optional[int] = None
    ) -> None:
        metrics = {}
        metrics["step"] = step
        metrics.update(metrics_dict)

        self.metrics.append(metrics)

        if self.tensorboard_writer is not None:
            for key in metrics_dict:
                self.tensorboard_writer.add_scalar(key, metrics_dict[key], step)

    def save(self) -> None:
        if not self.metrics:
            return

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        with open(
            os.path.join(self.output_dir, "training_log.csv"), "w", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=metrics_keys)
            writer.writeheader()
            writer.writerows(self.metrics)
