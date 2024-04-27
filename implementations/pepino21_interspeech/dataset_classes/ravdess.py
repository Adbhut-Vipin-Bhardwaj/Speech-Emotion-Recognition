"""
This file contains dataset class for loading the RAVDESS dataset (only audio).
This file also contains some transforms that can be applied on the dataset.

References
----------
[1] Livingstone SR, Russo FA (2018)
    The Ryerson Audio-Visual Database of Emotional Speech and Song
    (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in
    North American English.
    PLoS ONE 13(5): e0196391.
    https://doi.org/10.1371/journal.pone.0196391.
[2] https://zenodo.org/record/1188976
"""
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from pathlib import Path


class MergeCalmAndNeutralRAVDESS(object):
    """
    To merge the calm and neutral labels in RAVDESS.
    After this transform, label to emotion mapping
    will be as follows:
        0 -> calm/neutral
        1 -> happy
        2 -> sad
        3 -> angry
        4 -> fearful
        5 -> disgust
        6 -> surprised
    """
    def __init__(self):
        pass

    def __call__(self, emotion):
        new_emotion = emotion - 2
        if new_emotion == -1:
            new_emotion = 0
        return new_emotion


class MyResample(object):
    """
    Custom resampler, so that required change is made to both samp_rate and
    utterance
    """
    def __init__(self, in_samp_rate, out_samp_rate):
        self.in_samp_rate = in_samp_rate
        self.out_samp_rate = out_samp_rate
        self.transform = T.Resample(self.in_samp_rate, self.out_samp_rate)

    def __call__(self, utterance, in_samp_rate):
        if in_samp_rate == self.out_samp_rate:
            return utterance, in_samp_rate
        if in_samp_rate != self.in_samp_rate:
            raise ValueError("in_samp_rate of input != self.in_samp_rate \
                of MyResample")
        out_utterance = self.transform(utterance)
        return out_utterance, self.out_samp_rate


class RavdessAudio(Dataset):
    """
    RAVDESS dataset, audio only.
    """

    def __init__(self,
                 dir_path,
                 csv_path,
                 transform=None,
                 target_transform=None):
        """
        Parameters
        ----------
        dir_path : str
            path to the RAVDESS (e.g. "./foo/bar/")
        csv_path : str
            path to the csv file containing mapping from idx to
            file names
        transform : Optional[Callable]
            transform to be applied on the speech signal.
            call signature should be transform(speech_signal, sample_rate)
        target_transform : Optional[Callable]
            transform to be applied on the emotion label
        """
        self.dir_path = dir_path
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx, 1]
        file_name_list = (file_name.split('.')[0]).split('-')
        if file_name_list[1] == "01":
            # speech
            file_path = Path(self.dir_path)/"RAVDESS"\
                / "Audio_Speech_Actors_01-24"\
                / Path(f"Actor_{file_name_list[-1]}")\
                / Path(file_name)
        else:
            # song
            file_path = Path(self.dir_path)/"RAVDESS"\
                / "Audio_Song_Actors_01-24"\
                / Path(f"Actor_{file_name_list[-1]}")\
                / Path(file_name)
        emotion = int(file_name_list[2])
        spkr_id = int(file_name_list[-1])
        utterance, samp_rate = torchaudio.load(file_path)
        utterance = utterance[0,:]
        # some utterance arrays have size(0) = 2, but in all such cases,
        # utterance[0,:] = utterance[1,:]
        if self.transform:
            utterance, samp_rate = self.transform(utterance, samp_rate)
        if self.target_transform:
            emotion = self.target_transform(emotion)
        return utterance, utterance.size(0), samp_rate, spkr_id, emotion
        # (signal, length of signal (in samples), sample_rate, speaker id, label)
