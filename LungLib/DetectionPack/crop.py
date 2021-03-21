import skimage
from skimage.morphology import disk, dilation, remove_small_objects, erosion, closing, reconstruction, binary_closing, binary_dilation, binary_erosion
from skimage.measure import label, regionprops, perimeter
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import glob
import os
import nibabel as nib
import numpy as np
import cv2
import json
import os
import csv
import multiprocessing
from multiprocessing import Pool
import ntpath
from tqdm import tqdm

def get_segmented(images):
    #     segmented = np.zeros_like( images )
    binary_list = [get_segmented_lung(images[..., i])[1]
                   for i in range(0, images.shape[-1])]
    segmented = np.stack(binary_list, axis=-1)
    return segmented


def get_segmented_lung(image, th=-300):
    backup = image.copy()
    binary = backup < th
    selem = disk(1)
    dilation = binary_dilation(binary, selem)
    cleared = clear_border(dilation)
    label_image = label(cleared)
    areas = regionprops(label_image)
    areas.sort(key=lambda x: x.area, reverse=True)

    for area in areas[:2]:
        for r, c in area.coords:
            label_image[r, c] = -1

    large_component = label_image == -1
    selem = disk(3)
    erode = binary_erosion(large_component, selem)
    selem = disk(10)
    closing = binary_closing(erode, selem)
    edges = roberts(closing)
    segmented_binary = ndi.binary_fill_holes(edges)
    backup[segmented_binary == False] = np.min(backup)
    segmented_image = backup
    return segmented_image, segmented_binary.astype(np.int)


def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax




class CropLung:


    def __init__(self, zoomed_nifti_path, json_crop_path, temp_crop_path, thread_num, **kwargs):
        """
        Lung crop to lower region of interesest for each lung nifti in folder

        Parameters
        ----------

        path : zoomed nifti folder 
        json_crop_path : output file location IE: ( /notebooks/aaa/crop.json )

        Returns
        -------
        Dictionary : dict of crop position for each file Key : file name, Value : dict {x1,x2,y1,y2,z1,z2}
                    IE: {
                        "1": {
                        "x1": 49,
                        "x2": 465,
                        "y1": 128,
                        "y2": 374,
                        "z1": 0,
                        "z2": 407
                    }}
        """
        self.temp_crop_path = temp_crop_path
        zoomedNiftiPath = zoomed_nifti_path
        name = glob.glob(os.path.join(zoomedNiftiPath,"*.nii.gz"))
        
        jr = {}

        #for t in name:
        #    croop(t, jr)

        with Pool(thread_num) as pool:
            r = list(tqdm(pool.imap(self.croop, name), total=len(name)))
        
        txt_res = glob.glob(os.path.join(self.temp_crop_path,'*.txt'))

        for x in txt_res:
            file_name = ntpath.basename(x)
            file_name = file_name[:file_name.rindex('.txt')]
            fs = open(x,'r')
            content = fs.readline().split(" ")
            jr[file_name] = {
            "x1": int(content[0]),
            "x2": int(content[1]),
            "y1": int(content[2]),
            "y2": int(content[3]),
            "z1": int(content[4]),
            "z2": int(content[5])}

        

        with open(json_crop_path, 'w') as we:
            json.dump(jr, we, indent=4)

    def croop(self, path):
        loaded_images = nib.load(path)
        images = loaded_images.get_fdata()
        images = np.rot90(images, 1)
        ct_mask = get_segmented(images)
        x = bbox2_3D(ct_mask)

        name = ntpath.basename(path)[:ntpath.basename(path).rindex(".nii.gz")]

        with open(os.path.join(self.temp_crop_path,name+'.txt'),'w') as txtout:
            txtout.write(str(x[2])+" "+str(x[3])+" "+str(x[0])+" "+str(x[1])+" "+str(x[4])+" "+str(x[5]))
