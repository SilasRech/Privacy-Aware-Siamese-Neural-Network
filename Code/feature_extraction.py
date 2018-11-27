"""
Extract features from Audiosignals using Pythong speech features library

"""
import python_speech_features as fs
import numpy as np
import pandas as pd
from parameter import parameters


def feature_extraction(audiofile):
    """
    Extract features with settings as in parameters setup
    :param audiofile: Audiofile as numpy array
    :return: matrics of features (frames by features)
    """
    features_df = parameters('mfcc')

    batch_size = features_df.iloc[0]['length']
    batch_shift = features_df.iloc[0]['shift']
    name = features_df.iloc[0]['mode']
    num_features = features_df.iloc[0]['features']

    rate = 16000
    winlen = batch_size/16000
    winstep = batch_shift/16000

    if name == "MFCC":

        one_feature = fs.mfcc(audiofile, winlen=winlen, samplerate=rate, winstep=winstep, numcep=num_features, nfilt=64,
                              nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=False,
                              winfunc=np.hamming)

        num_rows = np.size(one_feature, 0)-1
        num_rows = num_rows - num_rows % num_features
        one_feature = one_feature[1:num_rows+1, :num_features]

    elif name == "MEL":

        one_feature = fs.logfbank(audiofile, winlen=winlen, samplerate=rate, winstep=winstep, nfilt=96, nfft=512,
                                  lowfreq=0, highfreq=None, preemph=0.97)

        num_rows = np.size(one_feature, 0) - 1
        num_rows = num_rows - num_rows % num_features
        one_feature = one_feature[1:num_rows + 1, :num_features]

    else:
        print('Feature-Type not found')
    features = pd.DataFrame(one_feature)

    return features, num_rows


