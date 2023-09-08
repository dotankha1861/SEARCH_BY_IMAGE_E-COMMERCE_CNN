import os
import cv2
import shutil
import configs as cf
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

num_aug_data = 10

# load_data:
print('Writing dir', cf.stddata_path)
for folder in os.listdir(cf.orgdata_path):

    curr_path = os.path.join(cf.orgdata_path, folder)
    save_path = os.path.join(cf.stddata_path, folder)
    
    try:
        os.mkdir(save_path)
    except:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    
    print('Creating folder', folder + '...')

    for _folder in os.listdir(curr_path):
    
        _curr_path = os.path.join(curr_path, _folder)
        _save_path = os.path.join(save_path, _folder)
           
        os.mkdir(_save_path)

        print('   Creating folder', _folder + '...', end = ' ')

        for file in os.listdir(_curr_path):
            curr_file = os.path.join(_curr_path,file)
    
            image = cv2.imread(curr_file) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, dsize = cf.size, interpolation= cv2.INTER_LINEAR)	
            image = expand_dims(image, axis = 0)	

            # augment data 
            myImageGen = ImageDataGenerator(width_shift_range=[-5,5], height_shift_range=[-5,5], rotation_range=10, shear_range=10)
            gen = myImageGen.flow(image, batch_size=1)
    
            for i in range(num_aug_data):
                myBatch = gen.next()
    
                I = myBatch[0].astype('uint8')
                
                path_save = os.path.join(_save_path, file[:file.find('.')] + '_aug_' + str(i) + '_' + file[file.find('.'):])
                cv2.imwrite(path_save, I)
    
        print('OK!')   
print('Completed!')  

images = []

print('Reading dir', cf.stddata_path)
for folder in os.listdir(cf.stddata_path):

    curr_path = os.path.join(cf.stddata_path, folder)
    
    print('Reading folder', folder + '...')

    for _folder in os.listdir(curr_path):
    
        _curr_path = os.path.join(curr_path, _folder)
      
        print('   Reading folder', _folder + '...', end = ' ')

        for file in os.listdir(_curr_path):
            curr_file = os.path.join(_curr_path,file)
    
            image = cv2.imread(curr_file)
            images.append(image)

        print('OK!')     

import random
random.shuffle(images)

print('Writing dir', cf.incdata_path)
for folder in os.listdir(cf.stddata_path):
    
    curr_path = os.path.join(cf.stddata_path, folder)
    save_path = os.path.join(cf.incdata_path, folder)
    
    try:
        os.mkdir(save_path)
    except:
        shutil.rmtree(save_path)
        os.mkdir(save_path)

    print('Creating folder', folder + '...')

    for _folder in os.listdir(curr_path):  

        _save_path = os.path.join(save_path, _folder)

        os.mkdir(_save_path)
           
        print('   Creating folder', _folder + '...', end = ' ')
        
        for i in range(num_aug_data):

            path_save = os.path.join(_save_path, _folder + '_image_' + str(i) + '.jpg')
            cv2.imwrite(path_save, images[i])

        images = images[num_aug_data:]   

        print('OK!')

print('Completed!')          

