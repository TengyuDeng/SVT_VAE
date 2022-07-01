# Contain submodule
from .lm import get_language_model

from .reconst.CNNReconst import CNNReconst
from .reconst.LSTMUNet import LSTMUNet
from .reconst.LSTMUNet import LSTMUNetPitchOnly
from .reconst.ReconstUNet import ReconstUNet
from .reconst.ReconstToy import ReconstToy
from .inference.CRNN_melody import CRNN_melody
from .inference.CNNZ import CNNZ
from .inference.ConvTCN import ConvTCN
from .basic.CNN import ConvNN
from .basic.TCN import TemporalConvNet
from .basic.UNet import UNet
from .renderer.Renderer import Renderer

model_list = {
    "CNN": ConvNN,
    "TCN": TemporalConvNet,
    "CRNN_melody": CRNN_melody,
    "ConvTCN": ConvTCN,
    "CNNReconst": CNNReconst,
    "LSTMUNet": LSTMUNet,
    "LSTMUNet_pitch_only": LSTMUNetPitchOnly,
    "ReconstUNet": ReconstUNet,
    "ReconstToy": ReconstToy,
    "CNNZ": CNNZ,
    "UNet": UNet,
    "Renderer": Renderer,
}
def get_model(num_classes_pitch=129, **configs):
    
    model_type = configs.pop('name')

    Model = model_list[model_type]
    
    model = Model(
        **configs,
        )

    return model