import SimpleITK as sitk
from multiprocessing import Pool, cpu_count
#from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm
import numpy as np
import nibabel as nib
import os
import shutil
import cc3d
import warnings
warnings.filterwarnings("ignore")


def applycc3d(path):
    #print(path)
    load_nifti = nib.load(path)
    affine = load_nifti.affine
    image = load_nifti.get_fdata().astype(np.uint8)
    image[image>0] = 1
    ccimage = cc3d.connected_components(image, connectivity=26)
    new_images = nib.Nifti1Image(ccimage,affine)
    new_images.to_filename(path)


class ZoomNifti:
    """
    zoom nifti images in target folder to new isotropic space

    Parameters
    ----------
    input_nifti_path : path to target folder (String)
    zoomed_nifti_path : path to output folder (String)
    new_space : new space
    applycc3d : Apply connected component 3d after zoom images (boolean, Default is False)
    thread_num : number of workers (int)
    Returns
    -------
    
    """
    def __init__(self, input_nifti_path, zoomed_nifti_path, new_space, apply_cc3d = False, thread_num = 1, **kwargs):
        

        self.input_folder_name = input_nifti_path
        self.output_folder_name = zoomed_nifti_path
        self.LIDC_IDRI_NIFTI = self.input_folder_name
        self.new_space = new_space
        self.LIDC_IDRI_NIFTI_ZOOM = self.output_folder_name
        self.thread_num = thread_num
        self.apply_cc3d = apply_cc3d
        self.nifti_paths = sorted([os.path.join(self.LIDC_IDRI_NIFTI, p)
                          for p in os.listdir(self.LIDC_IDRI_NIFTI) if 'checkp' not in p])
        #print("Number of nifti files: %d files" % len(self.nifti_paths))

        if os.path.exists(self.LIDC_IDRI_NIFTI_ZOOM):
            shutil.rmtree(self.LIDC_IDRI_NIFTI_ZOOM)

        os.mkdir(self.LIDC_IDRI_NIFTI_ZOOM)

    def zoom(self, path ):

        load_nifti = nib.load(path)
        affine = load_nifti.affine
        images = load_nifti.get_fdata().astype(np.float64)
        header = load_nifti.header
        
        x_space, y_space, z_space = header['pixdim'][1:4]
        x_space, y_space, z_space
        
        zoom =  np.asarray( [ x_space, y_space, z_space ] ) / np.asarray( [self.new_space]*3 )

        if self.apply_cc3d:
            order = 1
        else:
            order = 3

        zoomed_image = ndimage.zoom(images, zoom, order=order, mode='nearest', cval=0.0, prefilter=True)
        round_image = zoomed_image.astype(np.int16)
        new_affine = affine.copy()
        for i in range(3):
            new_affine[i, i] = new_affine[i, i] / zoom[i]
        
        new_images = nib.Nifti1Image(round_image,new_affine)

        if '.gz' not in path:
            optional_gz = ".gz"
        else:
            optional_gz = ""

        dst_path = os.path.join( self.LIDC_IDRI_NIFTI_ZOOM, path.split( os.sep )[-1]+ optional_gz )
        new_images.to_filename(dst_path)

        if self.apply_cc3d:
            applycc3d(dst_path)
        

    def execute(self):
        '''Begin process'''
        with Pool(self.thread_num) as pool:
            r = list(tqdm(pool.imap(self.zoom, self.nifti_paths), total=len(self.nifti_paths)))
    
    

#testing
if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    rex = ZoomNifti('W:\\lung_script\\dataset\\LNDb\\LNDb_MASK_ORIGINAL',
    'W:\\lung_script\\dataset\\LNDb\\LNDb_KKKL',0.50,False,1)
    rex.execute()