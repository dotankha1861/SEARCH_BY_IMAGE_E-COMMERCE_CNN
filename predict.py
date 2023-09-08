import os
import cv2
import numpy as np
import configs as cf
from keras_preprocessing.sequence import pad_sequences

def predict(image, labels, is_last):
    image = cv2.resize(image, dsize = cf.size, interpolation= cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis = 0)
    image_pad = pad_sequences(image, maxlen = cf.size[1], dtype = 'float32')/255.0
    model = cf.load_model(labels)
    pred = model.predict(image_pad)
    dict_labels = np.load(cf.get_path_label(labels), allow_pickle=True)
    if is_last == False: 
        return dict_labels[np.argmax(pred)]
    else:
        return dict_labels, pred.tolist()[0]

def predict_final(image, num_level = 2):
    labels = []
    for i in range(0,num_level):
        if i < num_level - 1: 
            label = predict(image, labels, is_last = False)
            labels.append(label)
        else: 
            dict_labels, pred = predict(image, labels, is_last = True)
            key_sm = [(dict_labels[i],pred[i]) for i in range(len(pred))]
            return [x[0] for x in sorted(key_sm, key = lambda x: x[1], reverse=True)]

if __name__ == '__main__':
    image = cv2.imread("D:\\HTTM\\dataset_org\\cooker\\ncd01\\ncd01_2.jpg")
    print("Predict id:", predict_final(image, 2)) 
