import pywt
from scipy.special import entr
from scipy.stats import skew
import numpy as np

class PreProcessor:
    def __init__(self, wavelet):
        self.wavelet = wavelet
    
    def transform(self, data):
        # data = data[:, :, :128]
        decomposed = pywt.wavedec(data, self.wavelet, level=3)
        # approx holds the information from 0 - 32Hz, detail from 32 to 64Hz
        approx = decomposed[0]
        detail = decomposed[1]
        return approx, detail
    
    def extract_features(self, data):
        # print("data coming in", data.shape)
        minimum = np.min(data, axis=2)
        maximum = np.max(data, axis=2)
        mean = np.mean(data, axis=2)
        median = np.median(data, axis=2)
        sd = np.std(data, axis=2)
        variance = np.var(data, axis=2)
        skewness = skew(data, axis=2)
        # entropy = entr(data)
        # print("finished features")
        # print(entropy)

        features = np.array([minimum, maximum, mean, median, sd, variance, skewness])
        # print("feature shape then max of first:", features.shape, features[2][0][0])
        # print(features.shape)
        reshaped_features = features.reshape(data.shape[0], data.shape[1], features.shape[0])
        # print("finished features reshape", reshaped_features.shape, reshaped_features[0][0][2])
        result_data = np.concatenate((data[:, :, :-data.shape[2]], reshaped_features), axis=2)
        # print("result_shape", result_data.shape)
        # print("----")
        return reshaped_features
        
        # return data[:, :, features]
