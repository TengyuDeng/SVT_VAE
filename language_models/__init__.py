from .HSMM import HSMM

model_list = {
    "HSMM": HSMM,
}
def get_language_model(**configs):
    
    model_type = configs["name"]

    Model = model_list[model_type]
    
    model = Model(**configs)

    return model