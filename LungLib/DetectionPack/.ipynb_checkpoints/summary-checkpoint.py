import os
import glob
import json
import nibabel as nib
import numpy as np
import collections
import ntpath
import csv
import numpy.linalg as npl



current_path = os.path.dirname(os.path.abspath(__file__))
print(current_path)


#####


def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(min(boxAArea, boxBArea) - interArea)
    return iou


def remove_file(dir_path) -> None:
    for root, dirnames, files in os.walk(dir_path, topdown=False):
        for f in files:
            full_name = os.path.join(root, f)
            if os.path.getsize(full_name) == 0:
                os.remove(full_name)

        for dirname in dirnames:
            full_path = os.path.realpath(os.path.join(root, dirname))
            if not os.listdir(full_path):
                os.rmdir(full_path)


def eliminate_large_gap(json_in, gapsize):
    has_change = False
    json_fix = {}
    for name in json_in:
        # outer loop for each case
        int_id = name
        json_fix[name] = {}

        used_nodule = []

        for each_nodule in json_in[name]:
            # print(used_nodule)
            # inner loop for each nodule in each case
            # filterout
            if each_nodule in used_nodule:
                continue

            # test current candidate from old dict and test with others
            current_nodule = json_in[name][each_nodule]

            grouped = False

            for each_candidate in json_in[name]:
                # check used
                if each_candidate in used_nodule:
                    continue

                # check same name
                if each_nodule in each_candidate:
                    continue
                candidate_nodule = json_in[name][each_candidate]
                ###
                # print(current_nodule[-1])
                # test last frame z coor to first frame of candidate
                current_first_box = current_nodule[0]
                current_last_box = current_nodule[-1]
                candidate_first_box = candidate_nodule[0]
                candidate_last_box = candidate_nodule[-1]
                if abs(candidate_first_box[0] - current_last_box[0]) <= gapsize+1:
                    if bb_iou(current_last_box[1:], candidate_first_box[1:]) >= 0.5:

                        tc = current_nodule+candidate_nodule
                        json_fix[name][each_nodule] = tc
                        used_nodule.append(each_nodule)
                        used_nodule.append(each_candidate)
                        has_change = True
                        grouped = True
                        break

            if not grouped:
                # single
                json_fix[name][each_nodule] = json_in[name][each_nodule]
                used_nodule.append(each_nodule)

    return json_fix, has_change


detect_txt_path = current_path+'/boxes/*.txt'
path_to_temove = current_path+'/boxes/'

nifti_path = current_path+"/sphere_075/"
crop_pos = current_path+'/crop.json'

out_file_name = current_path+'\\bbox_pred_all_afster.json'


class Summary:

    def __init__(self, detection_result_path : str,
                 zoomed_nifti_path : str, 
                 json_crop_path : str, 
                 summary_json_path : str,
                 summary_center_csv_path : str, 
                 summary_confident_theshold : int = 0.50, 
                 summary_allow_gap : int = 4,
                 summary_min_nodule_length : int= 3,
                 **kwargs):
        """
        zoom nifti images in target folder to new isotropic space

        Parameters
        ----------
        detect_txt_path : path to folder of detect file  (String)
        zoomed_nifti_path : path to zoomed nifti folder (String)
        json_crop_pos : path to json crop position. (String)
        summary_json_path : output filr IE: /notebooks/aaa/out.json (String)
        summary_center_csv_path : .csv path to save nodule center in real-world coordination 
        summary_confident_theshold : min average confident (of every detected bbox) of nodule to be considered as nodule default=0.5
        summary_allow_gap : frame gap between nodule to be considered as same nodule if [overlap area / min(areaA, areaB) >= 0.5] default=4
        summary_min_nodule_length : min length (frame) of nodule to be considered as nodule. default=3
        Returns
        -------

        """
        self.detect_txt_path = os.path.join(detection_result_path,"boxes/*.txt")
        self.nifti_path = zoomed_nifti_path
        self.json_crop_pos = json_crop_path
        self.out_file_name = summary_json_path
        self.min_nodule_length = summary_min_nodule_length
        self.allow_gap = summary_allow_gap
        self.confident_theshold = summary_confident_theshold

        self.summary_center_csv_path = summary_center_csv_path

        self.json_result = None

    def save_real_world_center(self,first_column_name = "LNDbID"):

        if self.json_result == None or self.summary_center_csv_path == None:
            raise Exception("run exeute() first to generate result") 

        with open(self.summary_center_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([first_column_name,'x','y','z','Nodule'])

        for name in self.json_result:
            niff = nib.load(os.path.join(self.nifti_path,name)+".nii.gz")
            aff = niff.affine
            #int_id = str(int(name[name.index('-')+1:]))
            int_id = str(name)
            for nodule_name in self.json_result[name]:
                nodule = np.array(self.json_result[name][nodule_name])
                #print(nodule[:,0])
                min_z = min(nodule[:,0])
                max_z = max(nodule[:,0])
                min_x = min(nodule[:,1])
                max_x = max(nodule[:,3])
                min_y = min(nodule[:,2])
                max_y = max(nodule[:,4])
                
                confid = np.average([x[-1] for x in nodule if x[-1] > 0])
                center = [(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2]
                
                real_pt = nib.affines.apply_affine(aff,center)
                with open(self.summary_center_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([int_id, real_pt[0]*-1, real_pt[1]*-1, real_pt[2], confid])

    def execute(self):
        # remove empty file
        # remove_file(self.path_to_temove)
        name_list = glob.glob(self.detect_txt_path)

        # grouping box by image name
        name_dict = {}

        for x in name_list:
            # print(x)
            if os.stat(x).st_size == 0:
                continue

            tmp_lst = []
            file_name = ntpath.basename(x)
            key_name = file_name[:file_name.index("_frame")]

            if key_name not in name_dict:
                tmp_lst = []
                tmp_lst.append(x)
                name_dict[key_name] = tmp_lst
            else:
                name_dict[key_name].append(x)

        sum_ = 0
        for i in name_dict:
            sum_ += len(name_dict[i])

        # fix crop offset
        with open(self.json_crop_pos) as j:
            jr = json.load(j)

        bbox = {}

        # detected box are croped, transform to real position in full images

        for i in name_dict:
            tmp_slice_box_outer = []
            # zoomed version header

            header = nib.load(os.path.join(
                self.nifti_path, i)+".nii.gz").header['dim'][1:4]
            # print(header)

            for each_file in name_dict[i]:
                tmp_slice_box_inner = []
                opened = open(each_file, 'r')

                # print(each_file)
                for b_no, line in enumerate(opened):
                    tmp_bbox = [float(x) for x in line.split()[1:]]

                    tmp_bbox[1] += jr[i]['x1']
                    tmp_bbox[2] += header[1]-jr[i]['y2']
                    tmp_bbox[3] += jr[i]['x1']
                    tmp_bbox[4] += header[1]-jr[i]['y2']

                    # number of nodule in each frame ie. 0,1,2...
                    tmp_bbox_out = []
                    # first is frame number
                    tmp_bbox_out.append(
                        int(each_file[each_file.rindex('_')+1:each_file.rindex('.')])+jr[i]['z1'])

                    # second is nodule number
                    tmp_bbox_out.append(b_no)
                    for c in tmp_bbox:
                        tmp_bbox_out.append(c)

                    tmp_bbox = tmp_bbox_out

                    tmp_slice_box_inner.append(tmp_bbox)
                tmp_slice_box_outer.append(tmp_slice_box_inner)
                # insert case as a key (file name)
            #key_s = each_file[each_file.rindex('\\')+1:each_file.rindex('_frame_')]
            key_s = ntpath.basename(each_file)[:ntpath.basename(each_file).rindex('_frame_')]
            #print(key_s)

            bbox[key_s] = tmp_slice_box_outer

        # sorting frame
        for key in bbox:
            bbox[key] = sorted(bbox[key], key=lambda x: x[0])

        # bbox is dict of boxes each dict contain list of bbox

        new_group = []
        for case_bbox in bbox:
            case = bbox[case_bbox]

            # assign for each grouped bbox
            running_no = 1
            last_frame = []
            last_frame_gap = collections.deque(maxlen=5)
            slice_groupped = []

            for i, each_frame in enumerate(case):
                # print(each_frame)
                if i == 0:
                    for n in each_frame:
                        n.append(running_no)
                        running_no += 1
                    last_frame = each_frame
                    continue

                for box in each_frame:
                    used_last_frame_bbox = []
                    next_box = False
                    for Fbox in last_frame:
                        if Fbox in used_last_frame_bbox:
                            continue
                        if bb_iou(box[3:], Fbox[3:]) >= 0.5 and box[0] - Fbox[0] <= 5:

                            box.append(Fbox[-1])
                            used_last_frame_bbox.append(Fbox)
                            break
                        else:
                            box.append(running_no)
                            running_no += 1
                            break
                last_frame = each_frame

        # flatten
        for case_bbox in bbox:
            new_frame_contain = []
            for xc in bbox[case_bbox]:
                for bbb in xc:
                    new_frame_contain.append(bbb)
            bbox[case_bbox] = new_frame_contain

        for key in bbox:
            bbox[key] = sorted(bbox[key], key=lambda x: x[-1])

        original_bbox = {}

        #print("max nodule")
        for case_bbox in bbox:
            center_list = []
            max_frame_number_percase = 0
            bc = bbox[case_bbox]
            # find total detection
            for i in bc:
                if i[-1] > max_frame_number_percase:
                    max_frame_number_percase = i[-1]
                    # print(max_frame_number_percase)

            # find center for each detection and convert spacing
            #center = []
            center = []
            index_samebox = 0
            for number in range(1, max_frame_number_percase+1):
                tmp_box = []

                for fm in bc:
                    if fm[-1] == number:
                        tmp_box.append(fm)

                tmp_box = np.array(tmp_box)

                confid = sum(tmp_box[:, 2])/len(tmp_box)
                #print(confid)
                if confid <= self.confident_theshold:
                    continue

                for in_tmp_box in tmp_box:
                    center.append(in_tmp_box)

                index_samebox += 1

            # center.append([(min_x+max_x)/2.,(min_y+max_y)/2.,(min_z+max_z)/2,    confid])

            center_list.append(center)
            # print(center_list)
            # print(len(center_list[0]))
            # break

            original_bbox[case_bbox] = center_list[0]

        json_out = {}
        for name in original_bbox:
            niff = nib.load(os.path.join(self.nifti_path, name)+".nii.gz")
            aff = niff.affine
            int_id = name
            json_out[name] = {}
            for center in original_bbox[name]:
                
                real_pt = center[:3]
                real_pt2 = center[3:6]
                # print(center)

                # print(center)
                # print(real_pt)

                if str(int(center[7])) not in json_out[name]:
                    json_out[name][str(int(center[7]))] = []
                json_out[name][str(int(center[7]))].append(
                    [center[0], center[3], center[4], center[5], center[6], center[2]])

        #print("Before fix GAP:")
        #for ter in json_out:
        #    print(ter, ' nodule: ', len(json_out[ter]))

        # fix gap > 1

        json_fix = json_out

        # eliminated large gap

        while True:
            # until there is no change
            json_fix, changed = eliminate_large_gap(
                json_fix, gapsize=self.allow_gap)
            if not changed:
                break

        # filling gap
        import copy
        for name in json_fix:
            temp_case = json_fix[name]
            for center in temp_case:
                # case 1 slice
                if len(temp_case[center]) == 1:
                    #print("one slice")
                    continue
                # total slice is correctly corresponsding to slice number range
                # [0][0] is Z of first box
                # [-1][0] is Z of last box
                elif len(temp_case[center])-1 == int(temp_case[center][-1][0] - temp_case[center][0][0]):
                    # print("OK")
                    continue
                else:
                    constuct_list = {}
                    for i in range(int(temp_case[center][0][0]), int(temp_case[center][-1][0])+1):
                        # create list of z
                        constuct_list[str(i)] = [float(i)]
                    # fill what already there
                    for x in temp_case[center]:
                        # change list z if already has [Z and x1y1 and x2y2]
                        if len(x) != 0:
                            constuct_list[str(int(x[0]))] = x

                    # filling gap
                    copy_constuct_list = copy.deepcopy(constuct_list)
                    # ref from construct_list write created frame into copy of it.
                    gap = 1
                    last_box = constuct_list
                    last_box_bck = 0.0
                    for it, box in enumerate(constuct_list):

                        if len(constuct_list[box]) == 1:
                            gap += 1
                            continue
                        # back up for later lookback
                        if it == 0:
                            last_box_bck = int(constuct_list[box][0])
                        else:
                            last_box_bck = last_box
                        # always copy lastest box index
                        last_box = int(constuct_list[box][0])

                        # create frame
                        for i in range(int(last_box_bck)+1, int(last_box)):
                            start_box = constuct_list[str(last_box_bck)][1:]
                            end_box = constuct_list[str(last_box)][1:]
                            # calculate slope
                            slope = [(end_box[0] - start_box[0])/gap,
                                     (end_box[1] - start_box[1])/gap,
                                     (end_box[2] - start_box[2])/gap,
                                     (end_box[3] - start_box[3])/gap, ]
                            #print(i - int(last_box_bck))

                            # calculate box by adding slope to the start box

                            copy_constuct_list[str(i)] = [
                                constuct_list[str(i)][0],
                                start_box[0] + slope[0] *
                                (i - int(last_box_bck)),
                                start_box[1] + slope[1] *
                                (i - int(last_box_bck)),
                                start_box[2] + slope[2] *
                                (i - int(last_box_bck)),
                                start_box[3] + slope[3] *
                                (i - int(last_box_bck)),
                                -1.0
                            ]
                            # print(i,gap)

                        gap = 1

                    # debugging
                    # for box in constuct_list:
                    #    print(copy_constuct_list[box])

                    out_list = []
                    for key, value in copy_constuct_list.items():
                        temp = [key, value]
                        out_list.append(value)

                    json_fix[name][center] = out_list
                    # print(constuct_list)

        
        #FILTERING NODULE LENGTH
        for ter in json_fix:
            remove_list = []
            for nodule in json_fix[ter]:

                if len(json_fix[ter][nodule]) < self.min_nodule_length:
                    remove_list.append(nodule)
            for removing_nodule in remove_list:
                json_fix[ter].pop(removing_nodule)

        #FILTERING NODULE CONFIDENT (NOT INCLUDE MISSING FRAME)
        for ter in json_fix:
            remove_list = []
            for nodule in json_fix[ter]:
                confid = [x for x in json_fix[ter][nodule] if x[-1] > 0]
                nda = np.array(confid)
                if np.average(nda[:,-1]) < self.confident_theshold:
                    remove_list.append(nodule)
            for removing_nodule in remove_list:
                json_fix[ter].pop(removing_nodule)
                
        #REORDER RAMAINING NODULES
        new_ordered = {}
        for ter in json_fix:
            new_ordered[ter] = {}
            for nodule_order_index, nodule in enumerate(json_fix[ter], start=1):
                #re insert with sequence name
                new_ordered[ter][str(nodule_order_index)] = json_fix[ter][nodule]
        

        json_fix = new_ordered

        with open(self.out_file_name, 'w', newline='') as csvfile:
            json.dump(json_fix, csvfile, indent=4)
        
        self.json_result = json_fix


if __name__ == '__main__':
    r = Summary(detect_txt_path="W:\\PROJECT_SCOTT_FULL\\boxes\\*.txt", zoomed_nifti_path="W:\\PROJECT_SCOTT_FULL\\sphere_075", json_crop_pos="W:\\PROJECT_SCOTT_FULL\\crop.json",
                out_file_name="W:\\PROJECT_SCOTT_FULL\\ooooooooooooo.json", allow_gap=1, min_nodule_length=5)
    r.execute()
