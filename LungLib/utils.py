import os




def create_path(path_list):
    for x in path_list:
        if not os.path.exists(x):
            os.makedirs(x)