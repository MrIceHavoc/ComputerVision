#!/usr/bin/python

import os
import tarfile
import zipfile
import wrapt
import cv2
from ast import literal_eval
import numpy as np
import open3d as o3d

class utility(object):
    def check_directory(function):
        def check(path, *args, **kwargs):
            if os.path.isfile(path):
                if tarfile.is_tarfile(path):
                    if path.endswith("tar.gz"):
                        with tarfile.open(path, "r:gz") as tar_ref:
                            tar_ref.extractall(path[:-len(".tar.gz")] + '/')
                    elif path.endswith("tar"):
                        with tarfile.open(path, "r:") as tar_ref:
                            tar_ref.extractall(path[-len(".tar")] + '/')
                elif zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path, 'r') as zip_ref: 
                        zip_ref.extractall(path[:-len(".zip")] + '/')
            if not os.path.isfile(path) and not os.path.isdir(path):
                os.makedirs(path)       
            return(path, args, kwargs)
        return check(function)


    def check_extension(extensions):
        @wrapt.decorator
        def wrapper(function, instance, *args, **kwargs):
            assert "extension" in kwargs
            assert kwargs['extension'] in extensions
            return function(*args, **kwargs)
        return wrapper


    @check_directory
    def load_images_from_dir(path):
        img_list = []
        for file in os.listdir(path):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_list.append(img)
        if len(img_list) < 2:
            raise Exception("At least 2 images are required for the computations.")
        return img_list


    @check_directory
    def load_intrinsic_params_from_txt(path, filename):
        counter = 0
        result = ""
        with open("{}{}.txt".format(path, filename), 'r') as f:
            for line in f:
                counter += line.count('[')
                if counter > 0:
                    result += line.rstrip()
                    counter -= line.count(']')
                else:
                    next
        return np.array(literal_eval(result))


    @check_directory
    @check_extension(extensions=['jpg, jpeg, png, txt, numpy, ply'])
    def write_data_to_file(path, filename, data, extension):
        if extension == 'numpy':
            file_path = path + filename + '{}' + '.txt'
        else:
            file_path = path + filename + '{}' + '.' + extension 
        counter = 0
        while os.path.isfile(file_path.format(counter)):
            counter += 1

        if extension in ['jpg', 'jpeg', 'png']:
            cv2.imwrite(file_path.format(counter), data)
        elif extension == 'numpy':
            np.savetxt(file_path, data)
        elif extension == 'txt':
            with open(file_path.format(counter), 'w') as outfile:
                outfile.write(data)   
        elif extension == 'ply':
            o3d.io.write_point_cloud(file_path, data)