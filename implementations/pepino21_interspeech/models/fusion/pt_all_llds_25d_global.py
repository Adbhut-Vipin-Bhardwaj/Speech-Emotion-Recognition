"""
This file contains code for the fusion model given in [1].
This model uses 25 dimensional Low Level Descriptors (LLDs) of eGeMAPS ([2]) and
all 13 layers of wav2vec 2.0. Concatenation is used for merging as given in [1]
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
import torchaudio.pipelines as ta_pipelines
import opensmile
import numpy as np
import pandas as pd


class PreTrainedAllLayersAndLLDsGlobalNorm(nn.Module):
    """
    Model using both eGeMAPS and wav2vec 2.0. Uses global normalization.
    """
    def __init__(self,
                 max_seq_len,
                 dim1,
                 norm_means_w2v2=None,
                 norm_std_devs_w2v2=None,
                 norm_means_eGeMAPS=None,
                 norm_std_devs_eGeMAPS=None,):
        """
        dim1: number of neurons in the fc layer of eGeMAPS part
        """
        super(PreTrainedAllLayersAndLLDsGlobalNorm, self).__init__()

        self.max_seq_len = max_seq_len
        # max length of LLD and w2v2 sequence
        self.norm_means_w2v2 = norm_means_w2v2
        # mean for feature normalization, or None
        # shape should be broadcastable to (13, *, 768)
        self.norm_std_devs_w2v2 = norm_std_devs_w2v2
        # standard deviation for feature normalization, or None
        # shape should be broadcastable to (13, *, 768)
        self.norm_means_eGeMAPS = norm_means_eGeMAPS
        # mean for feature normalization, or None
        # shape should be broadcastable to (*, 25)
        self.norm_std_devs_eGeMAPS = norm_std_devs_eGeMAPS
        # standard deviation for feature normalization, or None
        # shape should be broadcastable to (*, 25)
        self.dim1 = dim1

        self.eGeMAPS_xtrctr = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors
        )

        self.wav2vec2_model = ta_pipelines.WAV2VEC2_BASE.get_model()
        self.wav2vec2_model.eval()  # wav2vec2 model in eval mode
        for param in self.wav2vec2_model.parameters():
            param.requires_grad = False
        # now do setup to get features and lengths for each utterance
        self.w2v2_encodings = torch.zeros(13, self.max_seq_len, 768)
        self.w2v2_seq_len = 0
        self.wav2vec2_model.encoder.feature_projection.projection\
            .register_forward_hook((self.setup_hook(0)))  # local encodings
        for i in range(12):
            self.wav2vec2_model.encoder.transformer.layers[i].final_layer_norm\
                .register_forward_hook((self.setup_hook(i+1)))
            # transformer layers

        self.wtd_avg = nn.Conv2d(
            in_channels=13,
            out_channels=1,
            kernel_size=(1, 1),
            bias=False
        )
        nn.init.constant_(self.wtd_avg.weight, 1)
        self.pt_wise_conv1d_1_w2v2 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(1, 768)
        )
        self.pt_wise_conv1d_1_eGeMAPS = nn.Conv2d(
            in_channels=1,
            out_channels=dim1,
            kernel_size=(1, 25)
        )
        self.pt_wise_conv1d_2 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(1, dim1+128)
        )
        self.dropout = nn.Dropout(p=0.2)
        self.out_layer = nn.Linear(in_features=128, out_features=7)

    def setup_hook(self, num):
        def hook(model, input, output):
            self.w2v2_seq_len = min(output.size(1), self.max_seq_len)
            self.w2v2_encodings[num:num+1, :self.w2v2_seq_len, :]\
                = output.detach()[:, :self.w2v2_seq_len, :]
        return hook

    def forward(self, X):
        utterances, num_samples, _, spkr_ids = X

        batch_size = utterances.size(0)
        w2v2_features = torch.zeros((batch_size, 13, self.max_seq_len, 768))
        eGeMAPS_features = torch.zeros((batch_size, self.max_seq_len, 25))
        lengths = torch.zeros((batch_size))

        with torch.no_grad():
            for i in range(batch_size):
                eGeMAPS_feature_df = self.eGeMAPS_xtrctr.process_signal(
                    utterances[i, :int(num_samples[i].item())].cpu().numpy(),
                    16000  # sample rate
                )
                eGeMAPS_feature_np = eGeMAPS_feature_df.to_numpy()
                eGeMAPS_seq_len = eGeMAPS_feature_np.shape[0]
                if eGeMAPS_seq_len%2 == 0:
                    eGeMAPS_feature_np = eGeMAPS_feature_np.reshape((-1,2,25)).mean(axis=1)
                else:
                    eGeMAPS_feature_np = np.vstack(
                        (
                            eGeMAPS_feature_np[:-1,:].reshape((-1,2,25)).mean(axis=1),
                            eGeMAPS_feature_np[-1:,:]
                        )
                    )
                eGeMAPS_seq_len = min(eGeMAPS_feature_np.shape[0], self.max_seq_len)
                eGeMAPS_features[i, :eGeMAPS_seq_len, :] = torch.from_numpy(
                    eGeMAPS_feature_np[:eGeMAPS_seq_len, :]
                )
                # now extract w2v2 embeddings
                _ = self.wav2vec2_model(
                    utterances[i:i+1, :int(num_samples[i].item())])
                w2v2_seq_len = self.w2v2_seq_len
                w2v2_features[i,:,:,:] = self.w2v2_encodings
                self.w2v2_encodings[:,:,:] = 0  # prep for next utterance
                if w2v2_seq_len > eGeMAPS_seq_len:
                    # repeat last frame of eGeMAPS features
                    eGeMAPS_features[i, eGeMAPS_seq_len:w2v2_seq_len, :]\
                        = eGeMAPS_features[i, eGeMAPS_seq_len-1:eGeMAPS_seq_len, :]
                    lengths[i] = w2v2_seq_len
                elif w2v2_seq_len < eGeMAPS_seq_len:
                    # repeat last frame of w2v2 features
                    w2v2_features[i, w2v2_seq_len:eGeMAPS_seq_len, :]\
                        = w2v2_features[i, w2v2_seq_len-1:w2v2_seq_len, :]
                    lengths[i] = eGeMAPS_seq_len
                else:
                    # both lengths are equal
                    lengths[i] = eGeMAPS_seq_len
                if self.norm_means_w2v2 is not None:
                    w2v2_features[i, :, :int(lengths[i]), :] = (
                            w2v2_features[i, :, :int(lengths[i]), :]
                            - self.norm_means_w2v2
                    )
                if self.norm_std_devs_w2v2 is not None:
                    w2v2_features[i, :, :int(lengths[i]), :] = (
                            w2v2_features[i, :, :int(lengths[i]), :]
                            / self.norm_std_devs_w2v2
                    )
                if self.norm_means_eGeMAPS is not None:
                    eGeMAPS_features[i, :int(lengths[i]), :] = (
                            eGeMAPS_features[i, :int(lengths[i]), :]
                            - self.norm_means_eGeMAPS
                    )
                if self.norm_std_devs_eGeMAPS is not None:
                    eGeMAPS_features[i, :int(lengths[i]), :] = (
                            eGeMAPS_features[i, :int(lengths[i]), :]
                            / self.norm_std_devs_eGeMAPS
                    )

        w2v2_act_1 = self.wtd_avg(w2v2_features)
        # w2v2_act_1 shape = (B,1,Tmax,768)
        w2v2_act_1 = self.pt_wise_conv1d_1_w2v2(w2v2_act_1)
        # w2v2_act_1 shape = (B,128,Tmax,1)
        w2v2_act_1 = F.relu(w2v2_act_1)
        w2v2_act_1 = torch.permute(w2v2_act_1, (0, 3, 2, 1))
        # w2v2_act_1 shape = (B,1,Tmax,128)
        w2v2_act_1 = self.dropout(w2v2_act_1)

        eGeMAPS_act_1 = eGeMAPS_features.unsqueeze(1)
        # eGeMAPS_act_1 shape = (B,1,Tmax,25)
        eGeMAPS_act_1 = self.pt_wise_conv1d_1_eGeMAPS(eGeMAPS_act_1)
        # eGeMAPS_act_1 shape = (B,self.dim1,Tmax,1)
        eGeMAPS_act_1 = F.relu(eGeMAPS_act_1)
        eGeMAPS_act_1 = torch.permute(eGeMAPS_act_1, (0, 3, 2, 1))
        # w2v2_act_1 shape = (B,1,Tmax,self.dim1)
        eGeMAPS_act_1 = self.dropout(eGeMAPS_act_1)

        act_1 = torch.cat((w2v2_act_1, eGeMAPS_act_1), dim=3)
        # act_1 shape = (B,1,Tmax,self.dim1+128)

        act_1 = self.pt_wise_conv1d_2(act_1)  # act_1 shape = (B,128,Tmax,1)
        act_1 = F.relu(act_1)  # act_1 shape = (B,128,Tmax,1)
        act_1 = torch.permute(act_1, (0, 3, 2, 1))
        # act_1 shape = (B,1,Tmax,128)
        act_1 = self.dropout(act_1)

        act_2 = torch.zeros(batch_size, 128)
        for i in range(batch_size):
            act_2[i, :] = F.avg_pool2d(
                act_1[i:i+1, :, :int(lengths[i].item()), :],
                (int(lengths[i].item()), 1)
            ).squeeze()

        act_2 = self.out_layer(act_2)

        return act_2
