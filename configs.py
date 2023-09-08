import os
from keras.models import model_from_json

# path dir
orgdata_path = os.path.join("./dataset_org")
stddata_path = os.path.join("./dataset_std")
incdata_path = os.path.join("./dataset_inc")
model_path   = os.path.join("./model")
label_path   = os.path.join("./label")

# normalized image
size = (64, 64)
nchanels = 3

def get_path_label(labels):
    return os.path.join(label_path,"le_" + "_".join(labels) + ".npy")

def get_path_model_json(labels):
    return os.path.join(model_path,"model_" + "_".join(labels) + ".json")

def get_path_model_h5(labels):
    return os.path.join(model_path,"weight_" + "_".join(labels) + ".h5")

def load_model(labels):
    json_file = open(get_path_model_json(labels), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(get_path_model_h5(labels))
    return model
