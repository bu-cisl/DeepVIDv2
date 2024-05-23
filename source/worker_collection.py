import os

import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from skimage import io
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from tqdm import tqdm

import source.callback_collection as cc
import source.loss_collection as lc


class BaseWorker(object):
    def __init__(
        self,
        dataset_obj,
        network_obj,
        args,
        dataset_val_obj=None,
    ):
        self.dataset = dataset_obj
        self.model = network_obj
        self.dataset_val = dataset_val_obj

        self.parse_args(args)

        self.init_device()
        self.init_dataloader()
        self.init_network()
        self.init_scaler()

        if self.mode == "train":
            self.init_loss()
            self.init_optimizer()
            self.init_scheduler()
            self.init_callback()

    def parse_args(self, args):
        raise NotImplementedError()

    def init_device(self):
        raise NotImplementedError()

    def init_dataloader(self):
        raise NotImplementedError()

    def init_network(self):
        raise NotImplementedError()

    def init_loss(self):
        raise NotImplementedError()

    def init_optimizer(self):
        raise NotImplementedError()

    def init_scheduler(self):
        pass

    def init_scaler(self):
        raise NotImplementedError()

    def init_callback(self):
        pass

    def train(self, epoch):
        raise NotImplementedError()

    def val(self):
        raise NotImplementedError()

    def inference(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class BaseVIDWorker(BaseWorker):
    def __init__(
        self,
        dataset_obj,
        network_obj,
        args,
        dataset_val_obj=None,
    ):
        super(BaseVIDWorker, self).__init__(
            dataset_obj, network_obj, args, dataset_val_obj
        )

    def parse_args(self, args):
        # worker params
        self.mode = args.mode
        assert self.mode in ("train", "inference")

        self.model_string = args.model_string
        self.output_dir = args.output_dir

        if self.mode == "train":
            self.epochs = args.num_times_through_data
            self.batch_per_step = args.batch_per_step

        # dataset params
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        if self.mode == "train":
            self.num_sample_val = args.num_sample_val
            self.sampler_val_type = args.sampler_val_type

        # scaler params
        self.use_amp = not args.no_amp

        if self.mode == "train":
            # loss params
            self.loss_type = args.loss_type
            self.loss_weight = {
                "consistency_loss": 1.0,
                "l1_reg": args.l1_reg,
                "l2_reg": args.l2_reg,
            }

            # optimizer params
            self.learning_rate = args.learning_rate

            # scheduler params
            self.scheduler_type = args.scheduler_type

            # callback params
            self.step_save_model = args.step_save_model
            self.step_save_image = args.step_save_image

        elif self.mode == "inference":
            self.model_path = args.model_path
            self.output_file = args.output_file
            self.output_dtype = args.output_dtype

    def init_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
        if self.device == "cuda":
            cudnn.benchmark = True

    def init_dataloader(self):
        if self.mode == "train":
            self.train_dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )

            if self.dataset_val is None:
                self.dataset_val = self.dataset

            if self.sampler_val_type == "SubsetRandomSampler":
                sampler_val = SubsetRandomSampler(
                    np.array(
                        len(self.dataset_val)
                        / self.num_sample_val
                        * np.arange(self.num_sample_val),
                        dtype=int,
                    )
                )
            elif self.sampler_val_type == "RandomSampler":
                sampler_val = RandomSampler(
                    self.dataset_val, num_samples=self.num_sample_val
                )
            else:
                raise NotImplementedError(
                    "Unsupported sampler type: {}".format(self.sampler_val_type)
                )

            self.val_dataloader = DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                sampler=sampler_val,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )

        elif self.mode == "inference":
            self.inference_dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )

    def init_network(self):
        self.model = self.model.to(self.device)
        if self.mode == "inference":
            self.state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.state["model"])

    def init_loss(self):
        self.criterion = {"consistency_loss": lc.loss_selector(self.loss_type)}

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0,
        )

    def init_scheduler(self):
        if self.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=200,
                threshold=1e-3,
                min_lr=1e-7,
                verbose=True,
            )
        else:
            NotImplementedError(
                "Unsupported scheduler type: {}".format(self.scheduler_type)
            )

    def init_scaler(self):
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def init_callback(self):
        self.callback_save_image = cc.CallbackSaveImage(
            self.dataset_val, self.model, self.device, self.output_dir
        )
        self.callback_save_model = cc.CallbackSaveModel(self.output_dir)
        self.logger = cc.Logger(self.output_dir)


class DeepVIDv2Worker(BaseVIDWorker):
    def __init__(
        self,
        dataset_obj,
        network_obj,
        args,
        dataset_val_obj=None,
    ):
        super(DeepVIDv2Worker, self).__init__(
            dataset_obj, network_obj, args, dataset_val_obj
        )

    def calculate_loss(self, train_batch):
        log_dict = dict()
        for key in self.loss_weight:
            log_dict[key] = 0.0
        loss = 0.0

        X, y = train_batch
        X, y = X.to(self.device), y.to(self.device)

        y_pred = self.model(X)

        for key in self.loss_weight:
            if self.loss_weight[key] > 0:
                if key == "consistency_loss":
                    local_loss = self.criterion[key](y, y_pred)
                elif key == "l1_reg":
                    local_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if "conv" in name:
                            local_loss += param.abs().sum()
                elif key == "l2_reg":
                    local_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if "conv" in name:
                            local_loss += param.square().sum()
                else:
                    raise NotImplementedError("Unsupported loss type: {}".format(key))
                local_loss_weighted = local_loss * self.loss_weight[key]
                loss += local_loss_weighted
                log_dict[key] = local_loss_weighted.item()

        return loss, log_dict

    def train(self, epoch):
        self.model.train()
        loader = tqdm(self.train_dataloader, total=len(self.train_dataloader))

        log_dict_step = dict()
        for key in self.loss_weight:
            log_dict_step[key] = 0.0

        for batch_idx, train_batch in enumerate(loader):
            self.optimizer.zero_grad(set_to_none=True)
            cum_batch_idx = epoch * len(self.train_dataloader) + batch_idx

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
            ):
                loss, log_dict_batch = self.calculate_loss(train_batch)
            for key in log_dict_batch:
                log_dict_step[key] += log_dict_batch[key]

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if cum_batch_idx % self.batch_per_step == 0 and cum_batch_idx > 0:
                step = int(cum_batch_idx / self.batch_per_step)

                for key in log_dict_step:
                    log_dict_step[key] /= self.batch_per_step

                val_loss = self.val()
                self.model.train()
                if self.scheduler_type == "ReduceLROnPlateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                log_dict_step["val_loss"] = val_loss
                log_dict_step["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                self.logger.update(log_dict_step, step)

                print(
                    "\t".join(
                        [
                            "Step: {:d} / {:d}".format(
                                step,
                                int(
                                    (len(self.train_dataloader) / self.batch_per_step)
                                    * self.epochs
                                ),
                            )
                        ]
                        + [
                            "{}: {:.4f}".format(key, log_dict_step[key])
                            for key in log_dict_step
                        ]
                    )
                )

                if step % self.step_save_image == 0 and step > 0:
                    self.callback_save_image.save(step)
                    self.model.train()
                    self.logger.save()

                if step % self.step_save_model == 0 and step > 0:
                    self.callback_save_model.update(self.model, self.optimizer)
                    self.callback_save_model.save(step)

        self.callback_save_model.update(self.model, self.optimizer)
        self.callback_save_model.save("epoch_{:d}".format(epoch))

    def val(self):
        self.model.eval()
        loader = self.val_dataloader

        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, val_batch in enumerate(loader):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    X, y = val_batch
                    X, y = X.to(self.device), y.to(self.device)

                    y_pred = self.model(X)

                    loss = self.criterion["consistency_loss"](y, y_pred)
                val_loss += loss.item()
        val_loss /= len(loader)

        return val_loss

    def inference(self):
        self.model.eval()
        loader = tqdm(self.inference_dataloader, total=len(self.inference_dataloader))

        indiv_shape = self.dataset.get_output_size()
        final_shape = [len(self.dataset)]
        final_shape.extend(indiv_shape)
        chunk_size = [1]
        chunk_size.extend(indiv_shape)

        trend = self.dataset.trend[
            self.dataset.pre_frame : (
                self.dataset.num_frames - self.dataset.post_frame - 1
            )
        ]

        local_mean, local_std = self.dataset.__get_norm_parameters__()

        with h5py.File(
            os.path.splitext(self.output_file)[0] + ".h5", "w"
        ) as file_handle:
            dset_out = file_handle.create_dataset(
                "data",
                shape=tuple(final_shape),
                chunks=tuple(chunk_size),
                dtype="float32",
            )

            with torch.no_grad():
                for batch_idx, test_batch in enumerate(loader):
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                    ):
                        X, y = test_batch
                        X, y = X.to(self.device), y.to(self.device)

                        y_pred = self.model(X)

                    y_pred = y_pred.cpu().detach().numpy()

                    local_size = y_pred.shape[0]

                    start = batch_idx * self.batch_size
                    end = batch_idx * self.batch_size + local_size

                    # restore normalization
                    y_pred = y_pred * local_std + local_mean

                    # restore detrend
                    y_pred = (
                        y_pred + trend[start:end, np.newaxis, np.newaxis, np.newaxis]
                    )

                    dset_out[start:end, :] = y_pred

            y_pred = np.squeeze(dset_out)
            y_pred = y_pred.astype(self.output_dtype)

            tiff_filename = os.path.splitext(self.output_file)[0] + ".tif"
            io.imsave(
                tiff_filename,
                y_pred,
                imagej=True,
                compression="zlib",
                check_contrast=False,
            )
            print("Saved inference image. ")

        os.remove(os.path.splitext(self.output_file)[0] + ".h5")

    def run(self):
        if self.mode == "train":
            for epoch in range(self.epochs):
                self.train(epoch)

            self.callback_save_model.update(self.model, self.optimizer)
            self.callback_save_model.save()
            self.logger.save()

        elif self.mode == "inference":
            self.inference()
