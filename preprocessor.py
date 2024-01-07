import pywt

class PreProcessor():
    def __init__(self, wavelet):
        self.wavelet = wavelet
    
    def process(self, data):
        approx_coefficients, detail_coefficients = pywt.dwt(data, self.wavelet)
        print(approx_coefficients.shape, detail_coefficients.shape)