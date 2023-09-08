from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import configs as cf
import pandas as pd
import numpy as np
import os
import cv2

def load_data(typePro, path):

    X = []
    y = []

    labels = np.load(cf.get_path_label(typePro), allow_pickle=True).tolist()

    data_folder = os.path.join(path, typePro[0])
    for folder in os.listdir(data_folder):
    
        if typePro[0] == '':
            curr_path = os.path.join(data_folder, folder)
        else:
            curr_path = data_folder  

        for _folder in os.listdir(curr_path):
            _curr_path = os.path.join(curr_path, _folder)
     
            for file in os.listdir(_curr_path):
                curr_file = os.path.join(_curr_path,file)

                image = cv2.imread(curr_file)

                X.append(image)
                
                if typePro[0] == '':
                    y.append(labels.index(folder))
                else:
                    y.append(labels.index(_folder))
          
        if typePro[0] != '':
            break
    
    X = pad_sequences(X, maxlen = cf.size[1],dtype = 'float32')/255.0        
    y = to_categorical(y, len(labels))    
    
    return X, y

def Evaluate():

    Pre_Products = [['']]
    Pre_Products.extend([[product] for product in os.listdir(cf.stddata_path)])
    
    df = pd.DataFrame(index = ['std_dataset', 'inc_dataset'], 
                      columns= ['loss', 'accuracy'])

    for product in Pre_Products:

        model = cf.load_model(product)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        
        X_std, y_std = load_data(product, cf.stddata_path)
        X_inc, y_inc = load_data(product, cf.incdata_path)      
        
        loss_std, acc_std = model.evaluate(X_std, y_std)
        loss_inc, acc_inc = model.evaluate(X_inc, y_inc)
        
        df['loss'] = [loss_std, loss_inc]
        df['accuracy'] = [acc_std, acc_inc]

        if product[0] == '':
            print('Evaluate model predict typePro: ')
        else:
            print('Evaluate model predict idPro<%s>: '%(product[0]))

        print(df, end = '\n\n')

if __name__ == '__main__':
    Evaluate()