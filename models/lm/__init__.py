from .HSMM import HSMM
from .Vanila import Vanila

model_list = {
    "HSMM": HSMM,
    "Vanila": Vanila,
}
def get_language_model(**configs):
    
    model_type = configs["name"]

    Model = model_list[model_type]
    
    model = Model(**configs)

    return model