import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow as tf
import glob
import nibabel as nib
import ntpath
import cc3d
import cv2
from tqdm import tqdm
class Lungsegmentation:
    def execute(self):
        self.model = load_model(self.lung_segmentation_model_path, custom_objects={"dice_loss":dice_loss,"dice":dice})
        self.gen_lung_mask()
    
    def __init__(self, input_nifti_path,lung_segmentation_result_path,lung_segmentation_model_path, **kwargs):
        self.input_nifti_path = input_nifti_path
        self.lung_segmentation_result_path = lung_segmentation_result_path
        self.lung_segmentation_model_path = lung_segmentation_model_path
        
        
        

    def gen_lung_mask(self):

        all_file = glob.glob(os.path.join(self.input_nifti_path,"*.nii*"))
        #print(all_file)
        #for i in all_file:
        for i in tqdm(all_file):
            file_name=i
            image,header,affine = loadimage(i)
            name = ntpath.basename(i)
            label = genmask(self.model,image,name)
            #print(label.shape)
            #print(image.shape)
            if(".nii.gz" in name):
                out_label = os.path.join(self.lung_segmentation_result_path,"lung_mask_{}".format(name))

            else:
                out_label = os.path.join(self.lung_segmentation_result_path,"lung_mask_{}".format(name.replace(".nii",".nii.gz")))
            #print(out_label)
            connectivity = 8
            label = cc3d.connected_components(label,connectivity=26)
            len_cc = np.unique(label)
            pixel_list=[]
            name_list=[]

            for i in range(1,len(len_cc)):
                x,y,z = np.where(label == i)
                pixel_list.append(len(x))
                name_list.append(i)

            zipped = zip(pixel_list,name_list)
            max_volume = max(pixel_list)
            sorted_all_list = sorted(zipped, key= lambda x:x[0],reverse=True)
            for i,j in sorted_all_list:
                if(max_volume<=i*5):
                    g=1
                else:
                    label[label==j]=0
            label[label!=0]=1
            label = nib.Nifti1Image(label, affine, header)
            label.to_filename(out_label)
def loadimage(path):
    loaded_images = nib.load(path)
    affine = loaded_images.affine
    images = loaded_images.get_fdata()
    header = loaded_images.header
    
    images = np.rot90(images,1)
    img_nor = images.astype(np.float32)
    img_nor = img_nor + 1300
    img_nor = np.where(img_nor < 0.0, 0.0, img_nor)
    img_nor = (img_nor / 1600)
    img_nor = np.where(img_nor > 1.0, 1.0, img_nor)
    return img_nor,header,affine
def genmask(model,image,name):
    #print("asdfasd--------------fsef")
    height = 336
    width = 336
    img_channels=3
    #model.summary()
    out_array = np.zeros((image.shape[0],image.shape[1],image.shape[2]),dtype=np.uint8)
    #print(out_array.shape)
    counter=0
    for i in range(image.shape[2]):
        img = cv2.resize(image[:,:,i], (height,width), interpolation = cv2.INTER_AREA)
        img*=255.0
        img=img.astype(np.uint8)
        X = np.zeros((1,height, width, img_channels),dtype=np.uint8)
        X[0,:,:,0] = img
        X[0,:,:,1] = img
        X[0,:,:,2] = img
        
        counter+=1
        X=X/255.0
        pred_image = model.predict(X)
        pred_image*=255.0
        #cv2.imwrite(base_folder+"/test_image/{}.png".format(counter),pred_image[0])
        pred_image[pred_image<=127.]=0
        pred_image[pred_image>127.]=1
        pred_image = cv2.resize(pred_image[0], (image.shape[1],image.shape[0]), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite("/notebooks/VOLUME_sdb_5TB/_GIT/RUN_test_lung_seg/test_image/{}.png".format(counter),pred_image[0])
        out_array[:,:,i] = pred_image
        #print(type(out_array))
    out_array = np.rot90(out_array,-1)
    return out_array
def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))

  return 1 - numerator / denominator

def dice(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2)) + 0.0000000001

  return numerator / denominator