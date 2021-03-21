import tensorflow as tf
import glob
import numpy as np
import cv2
import json
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Model
import gc
import os
from os import walk
import ntpath
from tqdm import tqdm
def get_model_x(input_shape):
    
    
    inputs = Input(input_shape)

    X = Conv2D(filters=32, kernel_size=(5, 5),
               activation='relu', padding='same')(inputs)
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Conv2D(filters=64, kernel_size=(3, 3),
               activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = Conv2D(filters=128, kernel_size=(3, 3),
               activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    
    X = Conv2D(filters=256, kernel_size=(3, 3),
               activation='relu', padding='same')(X)

    X = Flatten()(X)

    X = Dense(1024, kernel_initializer="he_normal", activation='sigmoid')(X)
    X = Dense(512, kernel_initializer="he_normal", activation='sigmoid')(X)
    X = Dense(1,kernel_initializer="he_normal", activation='sigmoid')(X)

    model = Model(inputs=[inputs], outputs=[X])
    return model

def get_adaptive_lr_callback():
    from tensorflow.keras.callbacks import LearningRateScheduler

    def lr_scheduler(epoch, lr):
        decay_rate = 0.1
        decay_step = 50
        if epoch % decay_step == 0 and epoch > decay_step:
            return lr * decay_rate
        return lr

    adaptive_lr_callback = LearningRateScheduler(lr_scheduler)
    return adaptive_lr_callback

def compile_model(model, lr):
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow import keras
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.mean_squared_error,
                  metrics=[
                      keras.metrics.mean_absolute_error,
                      keras.metrics.mean_squared_error
                  ])
    return model


class CharacterPredict:
    def __init__(self, characteristic_image_path, characteristic_model_path, characteristic_output_path, **kwargs):
        """
        Predict characteristic of images in directory

        Parameters
        ----------

        characteristic_image_path : str
            Path to images directory
        characteristic_model_path : str
            Path to model Directory that contains model and multiplier
        characteristic_output_path : str
            Output file location IE: ( /notebooks/predicted.json )


        """
        self.characteristic_image_path = characteristic_image_path
        self.characteristic_output_path = characteristic_output_path
        self.characteristic_model_path = characteristic_model_path
    def execute(self):
        char_pic_path = self.characteristic_image_path
        model_path = self.characteristic_model_path
        _, case_folder, _ = next(walk(char_pic_path))

        pic_list = {}
        for x in case_folder:
            templ = glob.glob(os.path.join(char_pic_path, x, "*.png"))
            pic_list[x] = templ

        ref_name = []
        pic_a = []
        for case_name in pic_list:

            for pic_path in pic_list[case_name]:
                nodule_name = ntpath.basename(pic_path)[:-4]

                pic = cv2.imread(pic_path,0)
                gray = pic.reshape((pic.shape[0],pic.shape[1],1))
                pic_a.append(gray) 
                ref_name.append([case_name, nodule_name])

        pic = np.array(pic_a)
        pic = pic / 255.0

        mult_txt = glob.glob(os.path.join(model_path,"multiplier.txt"))
        if len(mult_txt) == 0:
            raise Exception("Stat multiplier file (multiplier.txt) does not exist.")

        temp_fs = open(mult_txt[0])
        max_stat = [float(f) for f in temp_fs.readline().split(" ")]
        #print(max_stat)
        #model must initiated by these Name and has .hdf5 format
        model_name_dict=['Subtlety*.hdf5',
                'Internal Structure*.hdf5',
                'Calcification*.hdf5',
                'Sphericity*.hdf5',
                'Margin*.hdf5',
                'Lobulation*.hdf5',
                'Spiculation*.hdf5',
                'Texture*.hdf5',
                'Malignancy*.hdf5']
        #model_list = glob.glob("model/*.hdf5")

        result = {}

        #iterating by model to reduce models load time
        for x, m_name in tqdm(enumerate(model_name_dict)):
            model = get_model_x((128,128,1))

            path_to_model = glob.glob(os.path.join(model_path,m_name))
            model.load_weights( path_to_model[0])

            char_res = model.predict(pic)
            char_res *= max_stat[x]


            for i,each_pic_res in enumerate(char_res):
                stat_name = m_name[:-6]
                case_name = ref_name[i][0]
                
                #print(self.characteristic_image_path+'/'+ref_name[i][0]+'/'+ref_name[1])
                nodule_id = ref_name[i][1]
                #added: id,x,y,z separated by '_' 
                first_block = ref_name[i][1].split('_')[0]
                nodule_id = first_block.split(',')[0]
                pos_x = first_block.split(',')[1]
                pos_y = first_block.split(',')[2]
                pos_z = first_block.split(',')[3]

                if case_name not in result:
                    result[case_name] = {}
                if nodule_id not in result[case_name]:
                    result[case_name][nodule_id] = {}
                #####
                    with open(os.path.join(self.characteristic_image_path,ref_name[i][0],str(nodule_id)+'.json')) as ldjson:
                        result[case_name][nodule_id] = json.load(ldjson)
                #####

                if 'Characteristic' not in result[case_name][nodule_id]:
                    result[case_name][nodule_id]['Characteristic'] = {}
                result[case_name][nodule_id]['Characteristic'] [stat_name] = float(char_res[i][0])
#                 result[case_name][nodule_id]['Position'] = {}
#                 result[case_name][nodule_id]['Position']['x'] = float(pos_x)
#                 result[case_name][nodule_id]['Position']['y'] = float(pos_y)
#                 result[case_name][nodule_id]['Position']['z'] = float(pos_z)
                
#                 first_block = ref_name[i][1].split('_')[1]
#                 pos_x = first_block.split(',')[0]
#                 pos_y = first_block.split(',')[1]
#                 pos_z = first_block.split(',')[2]
#                 result[case_name][nodule_id]['bbox3d'] = {}
#                 result[case_name][nodule_id]['bbox3d']['x0'] = int(pos_x)
#                 result[case_name][nodule_id]['bbox3d']['y0'] = int(pos_y)
#                 result[case_name][nodule_id]['bbox3d']['z0'] = int(pos_z)
                
#                 first_block = ref_name[i][1].split('_')[2]
#                 pos_x = first_block.split(',')[0]
#                 pos_y = first_block.split(',')[1]
#                 pos_z = first_block.split(',')[2]
#                 result[case_name][nodule_id]['bbox3d']['x1'] = int(pos_x)
#                 result[case_name][nodule_id]['bbox3d']['y1'] = int(pos_y)
#                 result[case_name][nodule_id]['bbox3d']['z1'] = int(pos_z)
                
#                 first_block = ref_name[i][1].split('_')[3]
#                 hu_vol = first_block.split(',')[0]
#                 hu_mean = first_block.split(',')[1]
#                 hu_median = first_block.split(',')[2]
#                 hu_min = first_block.split(',')[3]
#                 hu_max = first_block.split(',')[4]
#                 result[case_name][nodule_id]['properties'] = {}
#                 result[case_name][nodule_id]['properties']['nodule_volumn_mm3'] = float(hu_vol)
#                 result[case_name][nodule_id]['properties']['meanHU'] = float(hu_mean)
#                 result[case_name][nodule_id]['properties']['medianHU'] = float(hu_median)
#                 result[case_name][nodule_id]['properties']['minHU'] = float(hu_min)
#                 result[case_name][nodule_id]['properties']['maxHU'] = float(hu_max)
                
                
                
                #result[os.path.split(ppath[i])[-1]][namedict[x]] = float(char_res[i][0])

            del model
            gc.collect()

        with open(self.characteristic_output_path,'w') as js:
            json.dump(result,js,indent=4)

if __name__ == '__main__':
    tr = CharacterPredict("C:\\PROJECT_SCOTT_FULL\\CHAR\\pic","C:\\PROJECT_SCOTT_FULL\\CHAR\\model",
    "C:\\PROJECT_SCOTT_FULL\\CHAR\\output_py.json")
    tr.execute()
    
