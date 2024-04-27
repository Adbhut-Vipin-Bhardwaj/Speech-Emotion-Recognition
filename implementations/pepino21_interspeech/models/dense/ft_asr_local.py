"""
This file contains code for one of the models given in [1]. This model uses
wav2vec2 finetuned for ASR as feature extractor, uses the local encoding layer
of wav2vec2. Uses speaker normalization.

References:
    [1] Pepino, L., Riera, P., Ferrer, L.
    (2021) Emotion Recognition from Speech Using wav2vec 2.0 Embeddings.
    Proc. Interspeech 2021, 3400-3404, doi: 10.21437/Interspeech.2021-703
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.pipelines as ta_pipelines


class FineTunedAsrLocalLayer(nn.Module):
    """
    Model as given in [1], using only local encoder representations of
    fine tuned (for ASR) wav2vec2
    """
    def __init__(self,
                 max_seq_len,
                 norm_means=None,
                 norm_std_devs=None):
        super(FineTunedAsrLocalLayer, self).__init__()

        self.max_seq_len = max_seq_len
        # max length of wav2vec2 sequence
        self.norm_means = norm_means
        # dict of means for feature normalization, or None
        # shape should be broadcastable to (*, 768)
        self.norm_std_devs = norm_std_devs
        # dict of standard deviations for feature normalization, or None
        # shape should be broadcastable to (*, 768)

        self.wav2vec2_model = ta_pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        self.wav2vec2_model.eval()  # wav2vec2 model in eval mode
        for param in self.wav2vec2_model.parameters():
            param.requires_grad = False
        # now do setup to get features and lengths for each utterance
        self.encodings = torch.zeros(self.max_seq_len, 768)
        self.seq_len = 0
        self.wav2vec2_model.encoder.feature_projection.projection\
            .register_forward_hook((self.setup_hook()))  # local encodings

        self.pt_wise_conv1d_1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 768))
        self.pt_wise_conv1d_2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128))
        self.dropout = nn.Dropout(p=0.2)
        self.out_layer = nn.Linear(in_features=128, out_features=7)

    def setup_hook(self):
        def hook(model, input, output):
            self.seq_len = min(output.size(1), self.max_seq_len)
            self.encodings[:self.seq_len, :]\
                = output.detach()[0, :self.seq_len, :]
        return hook

    def forward(self, X):
        utterances, num_samples, _, spkr_ids = X

        batch_size = utterances.size(0)
        features = torch.zeros((batch_size, self.max_seq_len, 768))
        lengths = torch.zeros((batch_size))

        with torch.no_grad():
            for i in range(batch_size):
                _ = self.wav2vec2_model(
                    utterances[i:i+1, :int(num_samples[i].item())])
                features[i, :, :] = self.encodings
                lengths[i] = self.seq_len
                if self.norm_means is not None:
                    features[i, :int(lengths[i]), :] -= self.norm_means[spkr_ids[i].item()]
                if self.norm_std_devs is not None:
                    features[i, :int(lengths[i]), :] /= self.norm_std_devs[spkr_ids[i].item()]
                self.encodings[:, :] = 0


        act_1 = features.unsqueeze(1)  # act_1 shape = (B,1,Tmax,768)
        act_1 = self.pt_wise_conv1d_1(act_1)  # act_1 shape = (B,128,Tmax,1)
        act_1 = F.relu(act_1)  # act_1 shape = (B,128,Tmax,1)
        act_1 = torch.permute(act_1, (0, 3, 2, 1))
        # act_1 shape = (B,1,Tmax,128)
        act_1 = self.dropout(act_1)
        act_1 = self.pt_wise_conv1d_2(act_1)  # act_1 shape = (B,128,Tmax,1)
        act_1 = F.relu(act_1)  # act_1 shape = (B,128,Tmax,1)
        act_1 = torch.permute(act_1, (0, 3, 2, 1))
        # act_1 shape = (B,1,Tmax,128)
        act_1 = self.dropout(act_1)

        act_2 = torch.zeros(batch_size, 128)
        for i in range(batch_size):
            act_2[i, :] = F.avg_pool2d(
                act_1[i:i+1, :, :int(lengths[i].item()), :],
                (int(lengths[i].item()), 1)).squeeze()

        act_2 = self.out_layer(act_2)

        return act_2
