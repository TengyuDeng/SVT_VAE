# Contain submodule
from .lm import get_language_model

from .reconst.CNNReconst import CNNReconst
from .inference.CRNN_melody import CRNN_melody
from .inference.CNNZ import CNNZ
from .inference.ConvTCN import ConvTCN
from .basic.CNN import ConvNN
from .basic.TCN import TemporalConvNet

model_list = {
    "CNN": ConvNN,
    "TCN": TemporalConvNet,
    "CRNN_melody": CRNN_melody,
    "ConvTCN": ConvTCN,
    "CNNReconst": CNNReconst,
    "CNNZ": CNNZ,
}
def get_model(num_classes_pitch=129, **configs):
    
    model_type = configs["name"]

    Model = model_list[model_type]
    
    model = Model(
        num_classes_pitch=num_classes_pitch, 
        **configs,
        )

    return model