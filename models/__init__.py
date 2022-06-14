from .CRNN_melody import CRNN_melody
model_list = {
    "CRNN_melody": CRNN_melody,
}
def get_model(num_classes_pitch=129, **configs):
    
    model_type = configs["name"]

    Model = model_list[model_type]
    
    model = Model(
        num_classes_pitch=num_classes_pitch, 
        **configs,
        )

    return model