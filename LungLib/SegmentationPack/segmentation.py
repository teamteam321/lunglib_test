from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import json
import nibabel as nib
import glob
import numpy as np
import cv2
import os
from .nodule_diameter import axe_len

class Segmentation:
    def __init__(self, input_nifti_path, summary_json_path,segmentation_result_path,segmentation_model_path,characteristic_image_path, **kwargs):
        """
        Segmenting result from detection
        Parameters
        ----------
        input_nifti_path : str
            path to target folder
        summary_json_path : str
            path to summarized nodule position
        segmentation_result_path : str
            Output directory for nodules mask (.nii.gz)
        segmentation_model_path : str
            Segmentation model path
        characteristic_image_path : str
            Output directory for biggest bbox in each nodule
        Returns
        -------
        
        """
        self.input_nifti_path = input_nifti_path
        self.summary_json_path = summary_json_path
        self.segmentation_result_path = segmentation_result_path
        self.segmentation_model_path = segmentation_model_path
        self.characteristic_image_path = characteristic_image_path
        self.model = load_model(self.segmentation_model_path)
        #self.model = load_model(self.segmentation_model_path , custom_objects={"dice_loss":dice_loss,"dice":dice})
        with open(self.summary_json_path) as json_file:
            self.data = json.load(json_file)
        self.gen_and_pred()
    def gen_and_pred(self):
        num=0
        for i in self.data:
            filename = glob.glob(os.path.join(self.input_nifti_path , i)+".nii*")
            #print(">>>",len(filename))
           # print("len : ",len(filename))
            #print(">>>",filename)
            if not os.path.exists(os.path.join(self.characteristic_image_path,i)):
                os.makedirs(os.path.join(self.characteristic_image_path,i))

            image,header,aff,hu_image = load2(filename[0])
            space=header["pixdim"][1:4]
            #print(image.shape)
            nodule_numpy = np.zeros((image.shape[1],image.shape[0],image.shape[2]),dtype=np.uint8)
            new_header = header.copy()
            affine=aff
            for k in self.data[i]:
                width=128
                height=128
                img_channels=3
                duplist=[]
                characteristic_image=np.zeros((1,height, width, img_channels),dtype=np.uint8)
                biggest=0
                bcx=0
                bcy=0
                bcz=0
                
                nodule_confident_list = [] 
                
                for l in self.data[i][k]:
                    
                    if (l[5] != -1):
                        nodule_confident_list.append(l[5])
                        
                    
                    x1=int(round(l[1]/space[0]*0.75))
                    x2=int(round(l[3]/space[0]*0.75))
                    y1=int(round(l[2]/space[1]*0.75))
                    y2=int(round(l[4]/space[1]*0.75))
                    z=int(round(l[0]/space[2]*0.75))
                    nodule_area = (x2-x1)*(y2-y1)
                    cx=(x1+(x2-x1)/2)
                    cy=(y1+(y2-y1)/2)
                    #print(x2,x1,y2,y1)
                    #print(cx,cy)
                    
                    if(z in duplist):
                        continue
                    duplist.append(z)
                    imagecrop,pos,nore = crop2(image[:,:,z],x1,y1,x2,y2)
                    X = np.zeros((1,height, width, img_channels),dtype=np.uint8)
                    
                    X[0,:,:,0] = imagecrop
                    X[0,:,:,1] = imagecrop
                    X[0,:,:,2] = imagecrop
                    #check biggest area of nodule slice for characteristic
                    if nodule_area > biggest:
                        biggest = nodule_area
                        characteristic_image = X
                        bcx = cx
                        bcy = cy
                        bcz = z
                    X=X/255.0
                    
                    Y = np.zeros((1,height, width, 1),dtype=np.uint8)
                    Y[0,:,:,0]=imagecrop
                    Y=Y/255.0
                    #pred_image = self.model.predict(X)
                    pred_image = self.model.predict(Y)
                    pred_image*= 255.0
                    
                    pred_image[pred_image>127]=255
                    pred_image[pred_image<=127]=0
                    sizey=pos[1]-pos[0]
                    sizex=pos[3]-pos[2]
                    pred_image = cv2.resize(pred_image[0],(sizey,sizex), interpolation = cv2.INTER_NEAREST)
                    pred_image=cv2.transpose(pred_image)
                    nodule_numpy[pos[2]:pos[3],pos[0]:pos[1],z]=pred_image

                    num+=1
                nodule_numpy[nodule_numpy == 255] = int(k)
                
                #save additional infomation
                where_k = np.where(nodule_numpy == int(k))
                #print('total_pix',len(where_k[0]))
                
                #axeken (diameter ellipsiod)
                #print('------------------')
                axe_length,axe_vector = axe_len(where_k,space)
                
                
                nd_area = hu_image[where_k]
                #print(k,'vol: ',len(nd_area)*space[0]*space[1]*space[2])
                bbox_x0 = np.min(where_k[0])
                bbox_x1 = np.max(where_k[0])
                
                bbox_y0 = np.min(where_k[1])
                bbox_y1 = np.max(where_k[1])
                
                bbox_z0 = np.min(where_k[2])
                bbox_z1 = np.max(where_k[2])
                
                #print("centroids: ",np.mean(where_k[0]),np.mean(where_k[1]),np.mean(where_k[2]))
                centroid_x = np.mean(where_k[0])
                centroid_y = np.mean(where_k[1])
                centroid_z = np.mean(where_k[2])
                #print(k)
                
                #print(bbox_x0,bbox_x1)
                #print(bbox_y0,bbox_y1)
                #print(bbox_z0,bbox_z1)
                hu_volumn = np.round(len(nd_area)*space[0]*space[1]*space[2],4)
                hu_mean = np.round(np.mean(nd_area),4)
                hu_median = np.median(nd_area)
                hu_min = np.min(nd_area)
                hu_max = np.max(nd_area)
                
                #print(hu_volumn,hu_mean,hu_median,hu_min,hu_max)
                
                
                #print(np.unique(characteristic_image))
                #cv2.imwrite(os.path.join(self.characteristic_image_path,i,"{},{},{},{}_{},{},{}_{},{},{}_{},{},{},{},{}.png".format(k,centroid_x,centroid_y,centroid_z,bbox_x0,bbox_y0,bbox_z0,bbox_x1,bbox_y1,bbox_z1,hu_volumn,hu_mean,hu_median,hu_min,hu_max)),characteristic_image[0])
                cv2.imwrite(os.path.join(self.characteristic_image_path,i,"{},{},{},{}.png".format(k,centroid_x,centroid_y,centroid_z)),characteristic_image[0])
                # save .json stat with same name as the pic
                stat_out = {}
                
                stat_out['confident'] = sum(nodule_confident_list) / len(nodule_confident_list)
                
                stat_out['PCA'] = {}
                stat_out['PCA']["axe1_len_mm"] = axe_length[0]
                stat_out['PCA']["axe2_len_mm"] = axe_length[1]
                stat_out['PCA']["axe3_len_mm"] = axe_length[2]
                stat_out['PCA']['axe1_vector'] = list(axe_vector[0])
                stat_out['PCA']['axe2_vector'] = list(axe_vector[1])
                stat_out['PCA']['axe3_vector'] = list(axe_vector[2])
                
                stat_out['Position'] = {}
                stat_out['Position']['x'] = float(centroid_x)
                stat_out['Position']['y'] = float(centroid_y)
                stat_out['Position']['z'] = float(centroid_z)
                
                stat_out['bbox3d'] = {}
                stat_out['bbox3d']['x0'] = int(bbox_x0)
                stat_out['bbox3d']['y0'] = int(bbox_y0)
                stat_out['bbox3d']['z0'] = int(bbox_z0)
                stat_out['bbox3d']['x1'] = int(bbox_x1)
                stat_out['bbox3d']['y1'] = int(bbox_y1)
                stat_out['bbox3d']['z1'] = int(bbox_z1)
                
                stat_out['properties'] = {}
                stat_out['properties']['nodule_volumn_mm3'] = float(hu_volumn)
                stat_out['properties']['meanHU'] = float(hu_mean)
                stat_out['properties']['medianHU'] = float(hu_median)
                stat_out['properties']['minHU'] = float(hu_min)
                stat_out['properties']['maxHU'] = float(hu_max)
                
                with open(os.path.join(self.characteristic_image_path,i,str(k)+'.json'),'w') as json_stat_out:
                    
                    json.dump(stat_out,json_stat_out)
                
                
                
                
            new_images = nib.Nifti1Image(nodule_numpy, affine, new_header)
            new_images.to_filename(os.path.join(self.segmentation_result_path,i)+".nii.gz") 

def load2(name):
    file = glob.glob(name)
    #print("load name:",file)
    loaded_images = nib.load(file[0])
    images = loaded_images.get_fdata()
    affine = loaded_images.affine
    header = loaded_images.header
    images = np.rot90(images,1)
    img_nor = images.astype(np.float32)
    
    img_nor = img_nor + 1300
    img_nor = np.where(img_nor < 0.0, 0.0, img_nor)
    img_nor = (img_nor / 1600)
    img_nor = np.where(img_nor > 1.0, 1.0, img_nor)
    img_nor = cv2.flip(img_nor,0)
    return img_nor,header,affine,loaded_images.get_fdata() 
def crop2(img,x1,y1,x2,y2):
    ymax,xmax=img.shape
    if(x1<0):
        x2=x2-x1
        x1=0
    if(y1<0):
        y2=y2-y1
        y1=0
    if(x2>xmax):
        x1=x1-(x2-xmax)
        x2=xmax
    if(y2>ymax):
        y1=y1-(y2-ymax)
        y2=ymax
    if((x2-x1)>(y2-y1)):
        plus=int(((x2-x1)-(y2-y1))/2)
        y2=y2+plus
        y1=y1-plus
    elif((x2-x1)<(y2-y1)):
        plus=int(((y2-y1)-(x2-x1))/2)
        x2=x2+plus
        x1=x1-plus

    if((x2-x1)>(y2-y1)):
        y2=y2+1
    elif((x2-x1)<(y2-y1)):
        x2=x2+1
    cropimg = img[y1:y2, x1:x2]
    cropposition=[y1,y2,x1,x2]

    cropimg = (cropimg*255).astype(np.uint8)
    noresize = cropimg
    
    resizedcropimg = cv2.resize(cropimg, (128,128), interpolation = cv2.INTER_CUBIC)

    return resizedcropimg,cropposition,noresize
def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))

  return 1 - numerator / denominator

def dice(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
  denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2)) + 0.0000000001

  return numerator / denominator