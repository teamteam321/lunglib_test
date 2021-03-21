import os
import skimage
from skimage.morphology import disk, dilation, remove_small_objects,erosion, closing, reconstruction, binary_closing, binary_dilation ,binary_erosion
from skimage.measure import label,regionprops, perimeter
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import nibabel as nib
import numpy as np
import cv2
import json
import os
import csv
import glob
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm


class CreatePng:
    '''
    Take nifti in target folder and crop position of each nifti and ganerate picture (.png)
    pictures from each nifti will be in each subfolder

    Parameters
    ----------
    json_crop_path : dictionary of cropposition for each nifti (string path)
    zoomed_nifti_path : zoomed nifti folder path (string)
    thread_num : number of worker (int)
    
    '''
    def __init__(self, json_crop_path, zoomed_nifti_path, png_output_path, thread_num, **kwargs):
        self.CROPPOS = json_crop_path
        self.DRIVEPATH = zoomed_nifti_path
        self.PICPATH = png_output_path
        self.thread_nums = thread_num


    def lidc2fast(self,dict_crop):
        
        txtfile = dict_crop['name']

        name = txtfile
        #print(os.path.join(self.DRIVEPATH,name+".nii.gz"))
        file = glob.glob(os.path.join(self.DRIVEPATH,name+".nii.gz"))
        if len(file) == 0:
            return os.path.join(self.DRIVEPATH,name+".nii.gz")
        file = file[0]
        
        loaded_images = nib.load(file)
        images = loaded_images.get_fdata() 
        images = np.rot90(images,1)
        #crop
        images = images[dict_crop['y1']:dict_crop['y2']+1,dict_crop['x1']:dict_crop['x2']+1,dict_crop['z1']:dict_crop['z2']+1]
        #HU range
        img_nor = images.astype(np.float32)
        img_nor = img_nor + 1300.0
        img_nor = np.where(img_nor < 0.0, 0.0, img_nor)
        img_nor = (img_nor / 1600.0)
        img_nor = np.where(img_nor > 1.0, 1.0, img_nor)
        index = img_nor.shape[2]
        img_new = np.empty((img_nor.shape[0],img_nor.shape[1],3))

        #debug shape after crop
        #print(img_nor.shape)
            
        
        space = 0
        dis = space + 1
            
        for i2 in range(img_nor.shape[2]) :
                
                if i2 < dis:
                    img1 = img_nor[..., i2]*255
                else:
                    img1 = img_nor[..., i2-dis]*255
        
                img2 = img_nor[..., i2]*255
        
                if i2 >= index-dis:
                    img3 = img_nor[..., i2]*255
                else:
                    img3 = img_nor[..., i2+dis]*255
                
                
                img_new[:,:,0] = img1
                img_new[:,:,1] = img2
                img_new[:,:,2] = img3
                img_new = img_new.astype(np.uint8)
                

                fileName = name
                ext = ".png"

                #img_new = np.flipud(img_new)
                img_new = cv2.flip(img_new, 0)
                
                if not os.path.exists(os.path.join(self.PICPATH,fileName)):
                    os.makedirs(os.path.join(self.PICPATH,fileName))
                cv2.imwrite(os.path.join(self.PICPATH,fileName,fileName +"_frame_" + str(i2) + ext), img_new)
                                                      

    def execute(self):
        
        with open (self.CROPPOS) as j:
            jr = json.load(j)

        newlist = []
        for k in jr:
            jr[k]['name'] = k
            newlist.append(jr[k])
            
        
        processes=multiprocessing.cpu_count() - 4
        #print("Thread:"+str(processes))
        r = []
        with Pool(self.thread_nums) as pool:
            r = list(tqdm(pool.imap(self.lidc2fast, newlist), total=len(newlist)))
            
        for x in r:
            if x is not None:
                print("File not found: ",x)
           
     


# if __name__ == "__main__":
    
#     current_path = os.path.dirname(os.path.abspath(__file__))
#     print(current_path+"/sphere_075")
#     creatP = CreatePng(current_path+"/crop.json", current_path+"/sphere_075", current_path+"/png_test", 6)
#     creatP.execute()
    
#     #main()