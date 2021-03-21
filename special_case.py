import glob
import os
import json
import nibabel as nib
import glob
import numpy as np
import cv2
import os
write_data={}
with open('RUN_Scott_60_2/Characteristic.json') as f:
    data = json.load(f)
    
run_output_path = '/notebooks/VOLUME_sdb_5TB/_tmp/_GIT/RUN_Scott_60_2/'
seg_path = run_output_path + "segment_result"
s_case_path = "/notebooks/VOLUME_sdb_5TB/Special_nodule"
#niigz = glob.glob(os.path.join(niigz_path,"*.nii*"))
s_case = glob.glob(os.path.join(s_case_path,"*.nii*"))
def load(path):
    print(path)
    loaded_images = nib.load(path)
    images = loaded_images.get_fdata()
    affine = loaded_images.affine
    header = loaded_images.header
    images = np.rot90(images,1)
    return images,header,affine
s_case.sort()
for i in s_case:
    s_mask,h1,aff1 = load(i)
    name = i[i.index("segmentation-tumor")+19:i.index(".nii")-6]
    print(name)
    path = os.path.join(seg_path,"pre ablation {}.nii.gz".format(name))
    mask,h2,aff2 = load(path)
    print(mask.shape)
    print(s_mask.shape)
    if(mask.shape != s_mask.shape):
        print("Shape not equals")
        continue
    max_s_case = np.unique(s_mask.astype(np.uint8))
    write_data["pre ablation {}".format(name)]={}
    for j in max_s_case:
        if(j==0):
            continue
        x,y,z = np.where(s_mask == j)
        minx,miny,minz =min(x),min(y),min(z) 
        maxx,maxy,maxz =max(x),max(y),max(z)
        #print(minx,miny,minz,maxx,maxy,maxz)
        area = mask[minx:maxx,miny:maxy,minz:maxz]
        number_in_area =  np.unique(area)
        if(len(number_in_area) > 2):
            print("more than 1 element")
        
            
        np.sort(number_in_area)
        #print(number_in_area)
        #print(data["pre ablation {}".format(name)][str(number_in_area[1].astype(np.uint8))])
        print(number_in_area)
        if(len(number_in_area) !=1):
            
            detail_of_nodule=data["pre ablation {}".format(name)][str(number_in_area[1].astype(np.uint8))]
            write_data["pre ablation {}".format(name)][str(number_in_area[1].astype(np.uint8))] = detail_of_nodule
        #else:
            #write_data["pre ablation {}".format(name)][str(number_in_area[1].astype(np.uint8))] ={}
    with open(os.path.join(run_output_path,"special_nodule_characteristic.json"), 'w') as outfile:
        json.dump(write_data,outfile)