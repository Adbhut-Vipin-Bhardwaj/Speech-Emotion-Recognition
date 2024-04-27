"""
This file contains code for a model similar to the 'dense' model given in [1].
This model uses openSMILE toolkit as feature extractor to extract a sequence of
25 dimensional Low Level Descriptors (LLDs) of eGeMAPS ([2]).
Uses global normalization.

References:
    [1] Pepino, L., Riera, P., Ferrer, L.
    (2021) Emotion Recognition from Speech Using wav2vec 2.0 Embeddings.
    Proc. Interspeech 2021, 3400-3404, doi: 10.21437/Interspeech.2021-703
    [2] F. Eyben et al.
    "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS)
    for Voice Research and Affective Computing,".
    IEEE Transactions on Affective Computing, vol. 7, no. 2, pp. 190-202,
    1 April-June 2016, doi: 10.1109/TAFFC.2015.2457417.
"""
import torch
from torch import nn
import torch.nn.functional as F
import opensmile
import numpy as np
import pandas as pd


class LLDs25DimGlobalNorm(nn.Module):
    """
    Model using 25 dimensional LLDs as given in [2]. Uses speaker normalization.
    """
    def __init__(self,
                 max_seq_len,
                 dim1,
                 dim2,
                 norm_means=None,
                 norm_std_devs=None):
        """
        dim1: number of neurons in the first linear layer
        dim2: number of neurons in the second linear layer
        """
        super(LLDs25DimGlobalNorm, self).__init__()

        self.max_seq_len = max_seq_len
        # max length of LLD sequence
        self.norm_means = norm_means
        # mean for feature normalization, or None
        # shape should be broadcastable to (*, 25)
        self.norm_std_devs = norm_std_devs
        # standard deviation for feature normalization, or None
        # shape should be broadcastable to (*, 25)
        self.dim1 = dim1
        self.dim2 = dim2

        self.feature_xtrctr = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors
        )

        self.pt_wise_conv1d_1 = nn.Conv2d(in_channels=1, out_channels=dim1, kernel_size=(1, 25))
        self.pt_wise_conv1d_2 = nn.Conv2d(in_channels=1, out_channels=dim2, kernel_size=(1, dim1))
        self.dropout = nn.Dropout(p=0.2)
        self.out_layer = nn.Linear(in_features=dim2, out_features=7)

    def forward(self, X):
        utterances, num_samples, _, spkr_ids = X

        batch_size = utterances.size(0)
        features = torch.zeros((batch_size, self.max_seq_len, 25))
        lengths = torch.zeros((batch_size))

        with torch.no_grad():
            for i in range(batch_size):
                feature_df = self.feature_xtrctr.process_signal(
                    utterances[i, :int(num_samples[i].item())].cpu().numpy(),
                    16000  # sample rate
                )
                feature_np = feature_df.to_numpy()
                seq_len = feature_np.shape[0]
                if seq_len%2 == 0:
                    feature_np = feature_np.reshape((-1,2,25)).mean(axis=1)
                else:
                    feature_np = np.vstack(
                        (
                            feature_np[:-1,:].reshape((-1,2,25)).mean(axis=1),
                            feature_np[-1:,:]
                        )
                    )
                seq_len = min(feature_np.shape[0], self.max_seq_len)
                features[i, :seq_len, :] = torch.from_numpy(feature_np[:seq_len, :])
                lengths[i] = seq_len
                if self.norm_means is not None:
                    features[i, :int(lengths[i]), :] -= self.norm_means
                if self.norm_std_devs is not None:
                    features[i, :int(lengths[i]), :] /= self.norm_std_devs

        act_1 = features.unsqueeze(1)  # act_1 shape = (B,1,Tmax,25)
        act_1 = self.pt_wise_conv1d_1(act_1)  # act_1 shape = (B,self.dim1,Tmax,1)
        act_1 = F.relu(act_1)  # act_1 shape = (B,self.dim1,Tmax,1)
        act_1 = torch.permute(act_1, (0, 3, 2, 1))
        # act_1 shape = (B,1,Tmax,self.dim1)
        act_1 = self.dropout(act_1)
        act_1 = self.pt_wise_conv1d_2(act_1)  # act_1 shape = (B,self.dim2,Tmax,1)
        act_1 = F.relu(act_1)  # act_1 shape = (B,self.dim2,Tmax,1)
        act_1 = torch.permute(act_1, (0, 3, 2, 1))
        # act_1 shape = (B,1,Tmax,self.dim2)
        act_1 = self.dropout(act_1)

        act_2 = torch.zeros(batch_size, self.dim2)
        for i in range(batch_size):
            act_2[i, :] = F.avg_pool2d(
                act_1[i:i+1, :, :int(lengths[i].item()), :],
                (int(lengths[i].item()), 1)
            ).squeeze()

        act_2 = self.out_layer(act_2)

        return act_2
