# Contain submodule
from .lm import get_language_model

from .reconst import ReconstCNN, ReconstRNN

from .inference.CRNN_melody import CRNN_melody
from .inference.CRNN_lyrics import CRNN_lyrics

from .inference.CNNZ import CNNZ
from .inference.ConvTCN import ConvTCN

from .basic.CNN import ConvNN
from .basic.TCN import TemporalConvNet
from .basic.UNet import UNet

from .renderer.Renderer import Renderer

model_list = {
    "melody": {
        "CRNN_melody": CRNN_melody,
    },
    "lyrics": {
        "CRNN_lyrics": CRNN_lyrics,
    },
    "rendering": {
        "Renderer": Renderer,
    },
    "z_pre":{
        "CNN": CNNZ,
    },
    "reconst": {
        "CNN": ReconstCNN.CNN,
        "LSTMCNN": ReconstCNN.LSTMCNN,
        "LSTMCNNVariational": ReconstCNN.LSTMCNNVariational,
        "RNN": ReconstRNN.RNN,
        "RNNVariational": ReconstRNN.RNNVariational,
        "RNNVariationalNew": ReconstRNN.RNNVariationalNew,
        # "RNNVariationalNewFilterV1": ReconstRNN.RNNVariationalNewFilterV1,
        # "RNNVariationalNewFilterV2": ReconstRNN.RNNVariationalNewFilterV2,
    },
    # "CNN": ConvNN,
    # "TCN": TemporalConvNet,
    # "ConvTCN": ConvTCN,
    
    # "CNNZ": CNNZ,
    # "UNet": UNet,
    
}
def get_model(model_name="melody", **configs):
    
    model_type = configs.pop('name')
    Model = model_list[model_name][model_type]
    
    model = Model(
        **configs,
        )

    return model