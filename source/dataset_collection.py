import numpy as np
import torch
from skimage import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur

import source.network_collection as nc


class BaseDataset(Dataset):
    def __init__(self, args):
        super(BaseDataset, self).__init__()

        self.parse_args(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parse_args(self, args):
        raise NotImplementedError()

    def get_input_size(self):
        """
        This function returns the input size of the
        generator, excluding the batching dimension
        Parameters:
        None
        Returns:
        tuple: list of integer size of input array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[0]

        return local_obj.shape

    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension
        Parameters:
        None
        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]

        return local_obj.shape

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __get_norm_parameters__(self):
        raise NotImplementedError()


class VIDataset(BaseDataset):
    def __init__(self, args):
        super(VIDataset, self).__init__(args)

        self.imread()
        self.preprocess()

    def parse_args(self, args):
        self.noisy_data_path = args.noisy_data_path

    def imread(self):
        self.data = io.imread(self.noisy_data_path)
        self.data = self.data.astype("float32")

    def preprocess(self):
        self.detrend()
        self.normalize()
        self.num_frames, self.img_rows, self.img_cols = self.data.shape

    def detrend(self, order=2):
        trace = np.mean(self.data, axis=(1, 2))
        X = np.arange(1, trace.shape[0] + 1)
        X = X.reshape(X.shape[0], 1)
        pf = PolynomialFeatures(order)
        Xp = pf.fit_transform(X)
        md = LinearRegression()
        md.fit(Xp, trace)
        self.trend = md.predict(Xp)
        self.trend = self.trend.astype("float32")
        self.data -= np.reshape(self.trend, (self.trend.shape[0], 1, 1))

    def normalize(self):
        self.local_mean = np.mean(self.data)
        self.local_std = np.std(self.data)
        self.data = (self.data - self.local_mean) / self.local_std

    def __get_norm_parameters__(self):
        """
        This function returns the normalization parameters
        of the generator. This can potentially be different
        for each data sample
        Parameters:
        idx index of the sample
        Returns:
        local_mean
        local_std
        """
        local_mean = self.local_mean
        local_std = self.local_std

        return local_mean, local_std


class DIPDataset(VIDataset):
    def __init__(self, args):
        super(DIPDataset, self).__init__(args)

        self.epoch_index = 0

        self.list_samples = np.arange(
            self.pre_frame + self.pre_post_omission,
            self.num_frames - self.post_frame - self.pre_post_omission - 1,
        )

        self.mean_frame = np.mean(self.data, axis=0)
        self.std_frame = np.std(self.data, axis=0)

    def parse_args(self, args):
        super(DIPDataset, self).parse_args(args)

        self.pre_frame = args.input_pre_post_frame
        self.post_frame = args.input_pre_post_frame
        self.pre_post_omission = args.pre_post_omission

    def __len__(self):
        "Denotes the total number of batches"
        return len(self.list_samples)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            input: (pre_frame + post_frame, H, W)
            output: (1, H, W)

        """
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        # Generate indexes of the batch
        index_frame = self.list_samples[index]

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        sel = (input_index >= index_frame - self.pre_post_omission) & (
            input_index <= index_frame + self.pre_post_omission
        )
        input_index = input_index[~sel]

        # input
        data_img_input = self.data[input_index, :, :]

        # output
        data_img_output = self.data[np.newaxis, index_frame, :, :]

        input = data_img_input.astype("float32")
        output = data_img_output.astype("float32")

        input = torch.from_numpy(input)
        output = torch.from_numpy(output)

        return input, output


class N2VDataset(VIDataset):
    def __init__(self, args):
        super(N2VDataset, self).__init__(args)

    def parse_args(self, args):
        super(N2VDataset, self).parse_args(args)

        self.blind_pixel_ratio = args.blind_pixel_ratio
        assert self.blind_pixel_ratio >= 0
        self.blind_pixel_method = args.blind_pixel_method

    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension
        Parameters:
        None
        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]
        local_shape = list(local_obj.shape)
        local_shape[0] = 1

        return tuple(local_shape)

    def __len__(self):
        "Denotes the total number of batches"
        return self.num_frames

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            input: (1, H, W)
            output: (1 + 1(mask), H, W)

        """
        index_frame = index

        data_img_central_frame = self.data[index_frame, :, :]
        data_img_modified, mask = self.__generate_blind_image__(data_img_central_frame)

        # input
        data_img_input = data_img_modified[np.newaxis, :, :]

        # output
        data_img_output = np.concatenate(
            (
                data_img_central_frame[np.newaxis, :, :],
                mask[np.newaxis, :, :],
            ),
            axis=0,
        )

        input = data_img_input.astype("float32")
        output = data_img_output.astype("float32")

        input = torch.from_numpy(input)
        output = torch.from_numpy(output)

        return input, output

    def __generate_blind_image__(self, img):
        img_rows = img.shape[0]
        img_cols = img.shape[1]

        modified_img = np.copy(img)
        mask = np.zeros(img.shape)

        num_blind_pix = int(self.blind_pixel_ratio * img_rows * img_cols)
        indexes = np.arange(img_rows * img_cols - 1)
        np.random.shuffle(indexes)
        indexes_target = indexes[:num_blind_pix]
        idx_target_rows, idx_target_cols = np.unravel_index(
            indexes_target, shape=img.shape
        )

        if self.blind_pixel_method == "zeros":
            modified_img[idx_target_rows, idx_target_cols] = 0
        elif self.blind_pixel_method == "replace":
            indexes = np.arange(img_rows * img_cols - 1)
            np.random.shuffle(indexes)
            indexes_source = indexes[:num_blind_pix]
            idx_source_rows, idx_source_cols = np.unravel_index(
                indexes_source, shape=img.shape
            )
            modified_img[idx_target_rows, idx_target_cols] = img[
                idx_source_rows, idx_source_cols
            ]
        else:
            raise ValueError("Undefined blind pixel generation method. ")

        mask[idx_target_rows, idx_target_cols] = 1

        return modified_img, mask


class DeepVIDDataset(N2VDataset, DIPDataset):
    def __init__(self, args):
        super(DeepVIDDataset, self).__init__(args)

    def parse_args(self, args):
        super(DeepVIDDataset, self).parse_args(args)

    def get_output_size(self):
        """
        This function returns the output size of
        the generator, excluding the batching dimension
        Parameters:
        None
        Returns:
        tuple: list of integer size of output array,
        excluding the batching dimension
        """
        local_obj = self.__getitem__(0)[1]
        local_shape = list(local_obj.shape)
        local_shape[0] = 1

        return tuple(local_shape)

    def __len__(self):
        "Denotes the total number of batches"
        return len(self.list_samples)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            input: (pre_frame + post_frame + 1, H, W)
            output: (1 + 1(mask), H, W)

        """

        index_frame = self.list_samples[index]

        input_index = np.arange(
            index_frame - self.pre_frame - self.pre_post_omission,
            index_frame + self.post_frame + self.pre_post_omission + 1,
        )
        sel = (input_index >= index_frame - self.pre_post_omission) & (
            input_index <= index_frame + self.pre_post_omission
        )
        input_index = input_index[~sel]

        data_img_central_frame = self.data[index_frame, :, :]
        if self.blind_pixel_ratio > 0:
            data_img_modified, mask = self.__generate_blind_image__(
                data_img_central_frame
            )
        else:
            data_img_modified = data_img_central_frame
            mask = np.full_like(data_img_central_frame, 0)

        # input
        data_img_input = np.concatenate(
            (
                self.data[input_index, :, :],
                data_img_modified[np.newaxis, :, :],
            ),
            axis=0,
        )

        # output
        data_img_output = np.concatenate(
            (
                data_img_central_frame[np.newaxis, :, :],
                mask[np.newaxis, :, :],
            ),
            axis=0,
        )

        input = data_img_input.astype("float32")
        output = data_img_output.astype("float32")

        input = torch.from_numpy(input)
        output = torch.from_numpy(output)

        return input, output


class DeepVIDv2Dataset(DeepVIDDataset):
    def __init__(self, args):
        super(DeepVIDv2Dataset, self).__init__(args)

        self.gaussian = GaussianBlur(self.edge_gaussian_size, self.edge_gaussian_sigma)
        self.sobel = nc.SobelBlock()
        self.sobel = self.sobel.to(self.device)

        if self.edge_source_frame == "mean":
            self.edge_prior_global = self.extract_edge(self.mean_frame)
        elif self.edge_source_frame == "std":
            self.edge_prior_global = self.extract_edge(self.std_frame)
        else:
            raise ValueError("Undefined edge source frame type")

    def parse_args(self, args):
        super(DeepVIDv2Dataset, self).parse_args(args)

        self.edge_gaussian_size = args.edge_gaussian_size
        self.edge_gaussian_sigma = args.edge_gaussian_sigma
        self.edge_pre_post_frame = args.edge_pre_post_frame
        self.edge_source_frame = args.edge_source_frame

    def extract_edge(self, img: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract edges from an image.

        Parameters
        ----------
        img
        normalize

        Returns
        -------
        out
        """
        input = torch.tensor(img, device=self.device, dtype=torch.float)
        input = input.view(1, 1, input.shape[0], input.shape[1])
        out = self.sobel(self.gaussian(input))
        out = torch.squeeze(out).detach().cpu().numpy()

        if normalize:
            out = (out - np.mean(out)) / np.std(out)

        return out

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            input: (pre_frame + post_frame + 1 + 4, H, W)
            output: (1 + 1(mask), H, W)

        """
        input, output = super(DeepVIDv2Dataset, self).__getitem__(index=index)

        if self.edge_pre_post_frame is None:
            edge_prior = torch.from_numpy(self.edge_prior_global)
        else:
            index_frame = self.list_samples[index]

            if index_frame - self.edge_pre_post_frame < 0:
                edge_index = np.arange(
                    0,
                    0 + 2 * self.edge_pre_post_frame + 1,
                )
            elif index_frame + self.edge_pre_post_frame + 1 > self.num_frames - 1:
                edge_index = np.arange(
                    self.num_frames - 2 * self.edge_pre_post_frame - 2,
                    self.num_frames - 1,
                )
            else:
                edge_index = np.arange(
                    index_frame - self.edge_pre_post_frame,
                    index_frame + self.edge_pre_post_frame + 1,
                )

            if self.edge_source_frame == "mean":
                local_mean_frame = np.mean(self.data[edge_index, :, :], axis=0)
                edge_prior = self.extract_edge(local_mean_frame)
            elif self.edge_source_frame == "std":
                local_std_frame = np.std(self.data[edge_index, :, :], axis=0)
                edge_prior = self.extract_edge(local_std_frame)
            else:
                raise ValueError("Undefined edge source frame type")

            edge_prior = torch.from_numpy(edge_prior)

        input = torch.cat([input, edge_prior], dim=0)

        return input, output
