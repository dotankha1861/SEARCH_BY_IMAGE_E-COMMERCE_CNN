import os
import cv2
import numpy as np
import configs as cf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

# params loading data
'''
    level = 0  and type_sp = '': training product type (LEVEL 0)
    level = 1  and type_sp = '<product type>': training product id (LEVEL 1)  
'''
level = 1
type_sp = 'blender' # '', 'washing_machine', ...

# params traning
epochs = 100
batch_size = 128
_loss = 'categorical_crossentropy'
_optimizer = 'adam'
_metrics = ['accuracy']

# declare vars
labels = []
X = []
y = []

# load_data:
data_folder = os.path.join(cf.stddata_path, type_sp)
for folder in os.listdir(data_folder):

    if level == 0:
        labels.append(folder)
        curr_path = os.path.join(data_folder, folder)
    else:
        curr_path = data_folder  

    for _folder in os.listdir(curr_path):
        _curr_path = os.path.join(curr_path, _folder)

        if level == 1:
            labels.append(_folder)

        for file in os.listdir(_curr_path):
            curr_file = os.path.join(_curr_path,file)

            image = cv2.imread(curr_file)
            #image = cv2.resize(image, size, interpolation= cv2.INTER_LINEAR)

            X.append(image)

            if level == 0 :
                y.append(labels.index(folder))
            else:
                y.append(labels.index(_folder))
      
    if level == 1:
          break

# split_data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.2)

# limit length of sequences, scaler X in [0,1]
X_train = pad_sequences(X_train, maxlen = cf.size[1],dtype = 'float32')/255.0
X_val = pad_sequences(X_val, maxlen = X_train.shape[1],dtype = 'float32')/255.0
X_test = pad_sequences(X_test, maxlen = X_train.shape[1],dtype = 'float32')/255.0
num_classes = len(labels)

# one-hot vector
y_train = to_categorical(y_train, num_classes)    
y_val = to_categorical(y_val,num_classes)
y_test = to_categorical(y_test,num_classes)

# build model
model = Sequential()
input_size = (cf.size[0], cf.size[1], cf.nchanels)

model.add(Conv2D(256, kernel_size=(3, 3),activation='relu', input_shape= input_size))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Flatten())

model.add(Dropout(0.1))
model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.1))
model.add(Dense(512, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

# compile model
model.summary()
model.compile(loss = _loss, optimizer = _optimizer, metrics = _metrics)

# fit model
model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_val, y_val))

# save model
model.save(os.path.join(cf.model_path, 'weight_' + type_sp + ".h5"))
model_json = model.to_json()
with open(os.path.join(cf.model_path, 'model_' + type_sp + ".json"), 'w') as json_file:
    json_file.write(model_json)

# save labels
import numpy as np
np.save(os.path.join(cf.label_path, 'le_' + type_sp + '.npy'), labels, allow_pickle=True)           

# evaluate model
loss, acc = model.evaluate(X_test, y_test)
print("Loss: ",loss)
print("Acc: ",acc)