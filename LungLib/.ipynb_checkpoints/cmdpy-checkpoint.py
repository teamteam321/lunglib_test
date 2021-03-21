import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import glob
import json


from LungLib.DetectionPack import zoom, summary, crop, create
from LungLib.SegmentationPack import segmentation,lung_segmentation
from LungLib.CharacteristicPack import evalchar
from LungLib import utils

import multiprocessing


    
def run(config):
    with open(config) as json_file:
        load_kw = json.load(json_file)
        
    kw_args = {}
    ########################################################################
    
    #new spacing & total worker
    kw_args['new_space'] = load_kw['run_config']['new_space']
    kw_args['thread_num'] = load_kw['run_config']['thread_num']
    
    #nifti input path
    kw_args['input_nifti_path'] = load_kw['run_config']['input_nifti_path']
    #result folder
    kw_args['working_dir'] = load_kw['run_config']['working_dir']
    
    #path to efficientdet with custom infer function
    kw_args['efficientdet_path'] = load_kw['detection']['efficientdet_path']
    kw_args['efficientdet_name'] = load_kw['detection']['efficientdet_name']
    #path to saved model
    kw_args['efficientdet_saved_model_path'] = load_kw['detection']['efficientdet_saved_model_path']
    #min confident for nodule (average) 
    kw_args['summary_confident_theshold'] = load_kw['detection']['summary_confident_theshold']
    #max gap (frame) between bbox that will be merged
    kw_args['summary_allow_gap'] = load_kw['detection']['summary_allow_gap ']
    #min length for nodule
    kw_args['summary_min_nodule_length'] = load_kw['detection']['summary_min_nodule_length']
    
    
    #lung segmentation model path
    kw_args['lung_segmentation_model_path'] = load_kw['segmentation']['lung_segmentation_model_path']
    kw_args['segmentation_model_path'] = load_kw['segmentation']['segmentation_model_path']
    
    #characteristic models path
    kw_args['characteristic_model_path'] = load_kw['characteristic']['characteristic_model_path']
    
    ########################################################################
    
    
    
    utils.create_path([kw_args['working_dir']])
    #save original json
    with open(os.path.join(kw_args['working_dir'], 'config.json'), 'w') as outfile:
        json.dump(load_kw, outfile)
    kw_args['zoomed_nifti_path'] = os.path.join(kw_args['working_dir'], "_temp", "zoom_nifti")
    kw_args['json_crop_path'] = os.path.join(kw_args['working_dir'], "_temp", "crop")+".json"
    kw_args['temp_crop_path'] = os.path.join(kw_args['working_dir'], "_temp" , "crop_txt")
    kw_args['png_output_path'] = os.path.join(kw_args['working_dir'], "_temp" ,"png")
    kw_args['detection_result_path'] = os.path.join(kw_args['working_dir'], "detection_result")
    kw_args['summary_json_path'] = os.path.join(kw_args['working_dir'], "Detection_summary")+".json"
    kw_args['summary_center_csv_path'] = os.path.join(kw_args['working_dir'], "_temp", "Nodule_center")+".csv"
    kw_args['characteristic_image_path'] = os.path.join(kw_args['working_dir'], "characteristic_image")
    kw_args['characteristic_output_path'] = os.path.join(kw_args['working_dir'], "Characteristic")+".json"
    
    
    kw_args['lung_segmentation_result_path'] = os.path.join(kw_args['working_dir'], "lung_segment_result")
    kw_args['segmentation_result_path'] = os.path.join(kw_args['working_dir'], "segment_result")

    utils.create_path([kw_args['zoomed_nifti_path'], 
                 kw_args['png_output_path'], 
                 kw_args['segmentation_result_path'], 
                 kw_args['temp_crop_path'], 
                 kw_args['characteristic_image_path'],
                 kw_args['lung_segmentation_result_path']])


    
    
    print("Number of nifti files: %d files" % len(glob.glob(os.path.join(kw_args['input_nifti_path'],"*.nii*"))))
    print("Thread: ",kw_args['thread_num'])
    
    # gen lung masks for future usesage
    print("Generating lungmask...")
    lc = lung_segmentation.Lungsegmentation(**kw_args)
    # create new process to clear gpu mem after finished
    process_eval = multiprocessing.Process(target=lc.execute, args=())
    process_eval.start()
    process_eval.join()
    
    # zoom images to output path
    print("Preparing Images...")
    zoom.ZoomNifti(**kw_args).execute()

    #crop
    print("Cropping Lung...")
    crop.CropLung(**kw_args)

    #create png for inference
    print("Preparing Images for Detection...")
    create.CreatePng(**kw_args).execute()
    
    #run detection from script
    print("Detecting...")
    os.system('CUDA_VISIBLE_DEVICES="0" python '+ os.path.join(kw_args['efficientdet_path'],'model_inspect.py') + ' \
    --runmode=saved_model_infer \
    --model_name='+ kw_args['efficientdet_name'] +' \
    --saved_model_dir='+ kw_args['efficientdet_saved_model_path'] +' \
    --min_score_thresh='+ str(kw_args['summary_confident_theshold']) +' \
    --input_image='+ kw_args['png_output_path'] +'/**/*.png \
    --output_image_dir='+ kw_args['detection_result_path'] +' \
    --hparams="image_size=768x768,num_classes=1"')
    
    #summary detection result 
    print("Summarizing...")
    r = summary.Summary(**kw_args)
    r.execute()
    r.save_real_world_center()
   
    print("Creating Mask from result...")
    segmentation.Segmentation(**kw_args)
    
    print("Predicting Characteristic...")
    char = evalchar.CharacterPredict(**kw_args)
    char.execute()
    
#debug
if __name__ == '__main__':
    run('/notebooks/VOLUME_sdb_5TB/_GIT/config002.json')
    
    
    