#!/usr/bin/env python3
#!/usr/bin/env python2
#!/usr/bin/python

import wrapt
import numpy as np
import cv2
import os
from tabulate import tabulate
from ast import literal_eval
import open3d as o3d
import argparse
from itertools import permutations
import zipfile
import tarfile

np.set_printoptions(suppress=True)

# Default directories
RES_DIR = 'tasks_results/'
SG_RES_DIR = 'super_glue_results/'
SG_DIR = 'SuperGluePretrainedNetwork/'
SG_IMG_DIR = 'super_glue_images/'
IMG_DIR = 'stereo_images/'
INTRINSIC_FILE = 'intrinsic'

class KnnCountAction(argparse.Action):
    def __init__(self, option_strings, dest, help, type=int, default=2):
        super(KnnCountAction, self).__init__(option_strings, dest, help, type, default)
    def __call__(self, parser, namespace, values, option_string='k'):
        if values < 1:
            parser.error("Minimum value for {} is 1.").format(option_string)
        setattr(namespace, self.dest, values, option_string)

class CommandLine(argparse.ArgumentParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog="stereo_matcher", description="Stereo Matching Program.", prefix_chars=['-', '--'], epilog="Possible future improvement include adding more feature matching algorithms and allowing to compare results with other pretrained models besides SuperGlue.")
        self.parser.add_argument('--alg', dest='alg', help='Specifies the algorithm to be used for feature matching.', default='sift', choices=['sift', 'orb'])
        self.parser.add_argument('--img_input', dest='img_input', help='Directory or archive where input images are to be loaded from. Support PNG, JPG and JPEG formats.', default='stereo_images/')
        self.parser.add_argument('--intrinsic', dest='intrinsic', help="Directory where the files with intrinsic camera parameters are. Several files can be given as input.", nargs='*', action='append', default='intrinsic.txt')
        self.parser.add_argument('--output_dir', dest='output_dir', help="Directory where all the results will be saved to.", default='tasks_results/')
        self.parser.add_argument('--ratio', dest="ratio", help="Defines the value of Lowe's ratio to be used during feature matching. Default is 0.7.", type=float, default=0.7, choices=range(0.0, 1.0))
        self.parser.add_argument('-k', dest='k', help="Defines the of number of best matches found per each query descriptor. Default is 2 and value must be >= 1.", action=KnnCountAction, type=int, default=2)
        self.parser.add_argument('-viz', dest='viz', help="Defines if the obtained results are to be written to files. Defaut is true.", type=bool, action='store_true', choices=(True, False), default=True)
        self.parser.add_argument('-comp', dest='comp', help="Defines if the features obtained in task 1 are to be compared with learnable features from the SuperGlue model. Default is false.", type=bool, action='store_true', default=False)

    def parse_args(self):
        return self.parser.parse_args()

# Utility functions
class utility(object):
    def check_directory(function):
        def check(path, *args, **kwargs):
            if os.path.isfile(path):
                if tarfile.is_tarfile(path):
                    if path.endswith("tar.gz"):
                        with tarfile.open(path, "r:gz") as tar_ref:
                            tar_ref.extractall(IMG_DIR)
                    elif path.endswith("tar"):
                        with tarfile.open(path, "r:") as tar_ref:
                            tar_ref.extractall(IMG_DIR)
                elif zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path, 'r') as zip_ref: 
                        # TODO CHOOSE DIR TO UNZIP
                        zip_ref.extractall(IMG_DIR)
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


# Functions to obtain results from pretrained models
class pretrained_model(object):
    def obtain_results_from_super_glue(img_dir, sg_dir, out_dir, res_dir, max_length=1, outdoor=True, return_results=True):
        def load_super_glue_results_from_npz(directory):
            for file in os.listdir(directory):
                if not file.endswith('.npz'):
                    next
                else:
                    npz = np.load("{}{}".format(directory, file))
                    kp1 = npz['keypoints0'].shape[0]
                    kp2 = npz['keypoints1'].shape[0]
                    matches = np.sum(npz['matches'] > -1)
                    return (kp1, kp2, matches)
    
        # Obtain features using SuperGlue model from the given images
        weights = 'outdoor' if outdoor else 'indoor'
        os.system(f"cp -r {img_dir} {sg_dir + img_dir}")
        os.chdir(f'{sg_dir}')
        os.system(f"./match_pairs.py --input_dir='{img_dir}' --output_dir='{out_dir}' --max_length={max_length} --superglue='{weights}' --viz")
        os.system(f"rm -r {img_dir}")
        os.chdir('../')
        os.system(f'mv {sg_dir + out_dir} {res_dir + out_dir}')
        if return_results:
            return load_super_glue_results_from_npz(res_dir + out_dir)
        else:
            return
        

def task_1_feature_extraction(image_pair, alg='sift', ratio=0.7, k=2, viz=True, comp=False):
    if alg == 'sift':
        alg = cv2.SIFT_create()
        bfm = cv2.BFMatcher()
    elif alg == 'orb':
        alg = cv2.ORB_create()
        bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Find the keypoints and descriptors for both images using the SIFT algorithm
    kp1, desc1 = alg.detectAndCompute(image_pair[0], None)
    kp2, desc2 = alg.detectAndCompute(image_pair[1], None)

    # Obtain the best K feature matches from each query descriptor
    matches = bfm.knnMatch(desc1, desc2, k)

    good_matches1 = []
    
    # Apply relative feature matching ratio to filter out matches with values close to 2nd best match
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches1.append(m)
    
    # Apply the same feature matching process, but we shift the query and train images
    matches = bfm.knnMatch(desc2, desc1, k)
    
    good_matches2 = []
    
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches2.append(m)
      
    symmetric_matches = []
    
    # Filter out non-symmetric matches
    for gm1 in good_matches1:
        for gm2 in good_matches2:
            if gm1.trainIdx == gm2.queryIdx and gm1.queryIdx == gm2.trainIdx:
                symmetric_matches.append([gm1])         
    
    # Draw image with obtained matches and write it to a file
    if viz == True:
        match_img = cv2.drawMatchesKnn(image_pair[0], kp1, image_pair[1], kp2, symmetric_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        filename = "sift_matches"
        utility.write_data_to_file(RES_DIR, filename, match_img, extension='jpg')
        
    # Compare with results obtained from SuperGlue model and write to file
    if comp == True:
        sg_kp1, sg_kp2, sg_matches = pretrained_model.obtain_results_from_super_glue(SG_IMG_DIR, SG_DIR, SG_RES_DIR, RES_DIR)
        sift_ratio = len(symmetric_matches) / (len(kp1) + len(kp2))
        sg_ratio = sg_matches / (sg_kp1 + sg_kp2)
        results = [['Method', f'Keypoints Image 1', f'Keypoints Image 2', 'Matches', 'Matches/Keypoints'],
                ['SIFT', "{}".format(len(kp1)), "{}".format(len(kp2)), "{}".format(len(symmetric_matches)), "{}".format(sift_ratio)],
                ['SuperGlue', "{}".format(sg_kp1), "{}".format(sg_kp2), "{}".format(sg_matches), "{}".format(sg_ratio)]]
        table = tabulate(results, headers='firstrow', tablefmt='github')
        filename = 'SuperGlue_comparison'
        utility.write_data_to_file(RES_DIR, filename, table, extension='txt')
    
    # Return features corresponding to obtained matches
    features = []
    for sym_mat in sorted(symmetric_matches, key = lambda x: x[0].distance):
        mat = sym_mat[0]
        features.append({'kp1': kp1[mat.queryIdx], 'kp2': kp2[mat.trainIdx], 'desc1': desc1[mat.queryIdx], 'desc2': desc2[mat.trainIdx], 'match': mat})
    
    return features


def task_2_estimate_fundamental_matrix(image_pair, features, viz=True):
    def draw_epipolar_lines(image_pair, epi_geo):
        _, img1_width = image_pair[0].shape
        _, img2_width = image_pair[1].shape
        img1 = cv2.cvtColor(image_pair[0],cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(image_pair[1],cv2.COLOR_GRAY2BGR)
        
        for epi_elem in epi_geo:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x1_c,y1_c = map(int, [0, -epi_elem['line1'][2] / epi_elem['line1'][1]])
            x1, y1 = map(int, [img1_width, -(epi_elem['line1'][2] + epi_elem['line1'][0] * img1_width) / epi_elem['line1'][1]])
            img1 = cv2.line(img1, (x1_c, y1_c), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(epi_elem['inlier1']), 5, color, -1)
            
            x2_c,y2_c = map(int, [0, -epi_elem['line2'][2] / epi_elem['line2'][1]])
            x2, y2 = map(int, [img2_width, -(epi_elem['line2'][2] + epi_elem['line2'][0] * img2_width) / epi_elem['line2'][1]])
            img2 = cv2.line(img2, (x2_c, y2_c), (x2, y2), color, 1)
            img2 = cv2.circle(img2, tuple(epi_elem['inlier2']), 5, color, -1)
        return img1, img2
    
    # Obtain coordinate points from features to calculate fundamental matrix
    points_img1 = []
    points_img2 = []
    
    for feat in features:
        (x1, y1) = feat['kp1'].pt
        points_img1.append((x1, y1))
        (x2, y2) = feat['kp2'].pt
        points_img2.append((x2, y2))
    
    # Intermediate calculation to reject outliers using RANSAC
    points_img1 = np.int32(points_img1)
    points_img2 = np.int32(points_img2)
    _, mask = cv2.findFundamentalMat(points_img1, points_img2, cv2.FM_RANSAC)
    
    # Select inlier points
    points_img1 = points_img1[mask.ravel() == 1]
    points_img2 = points_img2[mask.ravel() == 1]
    
    # Compute fundamental matrix using 8-point algorithm
    F_matrix, _ = cv2.findFundamentalMat(points_img1, points_img2, cv2.FM_8POINT)
    if np.linalg.matrix_rank(F_matrix) != 2:
        raise Exception("Fundamental matrix must have rank 2.")


    # Compute epipolar lines from inliers 
    lines1 = cv2.computeCorrespondEpilines(points_img2.reshape(-1, 1, 2), 2, F_matrix)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(points_img1.reshape(-1, 1, 2), 1, F_matrix)
    lines2 = lines2.reshape(-1, 3)
    
    # Return epipolar geometry features
    epi_geo = []
    for point1, point2, line1, line2 in zip(points_img1, points_img2, lines1, lines2):
        epi_geo.append({'inlier1': point1, 'inlier2': point2, 'line1': line1, 'line2': line2})
    
    if viz == True:
        # Write fundamental matrix to text file
        filename_fmat = "fundamental_matrix"
        utility.write_data_to_file(RES_DIR, filename_fmat, F_matrix, extension='numpy')
        
        # Draw both images epipolar lines and write them to files
        img3, img4 = draw_epipolar_lines(image_pair, epi_geo)
        filename_e1 = "left_epipolar_lines"
        filename_e2 = "right_epipolar_lines"
        utility.write_data_to_file(RES_DIR, filename_e1, img3, extension='jpg')
        utility.write_data_to_file(RES_DIR, filename_e2, img4, extension='jpg')
    
    return (F_matrix, epi_geo)


def task_3_compute_essential_matrix(F_matrix, intrinsic_K, viz=True):
    # Compute essential matrix, with K_1 = K_2, since camera is the same
    F = np.matrix(F_matrix)
    K = np.matrix(intrinsic_K)
    E = np.transpose(K) @ F @ K
    if np.linalg.matrix_rank(E) != 2:
        raise Exception("Essential matrix must have rank 2.")
    
    # Enforce value constraint of diagonal matrix = diag(1, 1, 0)
    U, S, Vt = np.linalg.svd(E)
    D = np.diag([1, 1, 0])
    if not np.array_equal(D, np.diag(S)):
        E = U @ D @ Vt
    
    # Write essential matrix to text file
    if viz == True:
        filename = "essential_matrix"
        utility.write_data_to_file(RES_DIR, filename, E, extension='numpy')
    
    return E


def task_4_obtain_camera_pose(E_matrix, intrinsic_K, epi_geo, viz=True):
    def triangulation_and_pose_check(P, epi_geo):
        # Perform triangulation
        points3D = []
        for elem in epi_geo:
            point4D = cv2.triangulatePoints(P, P, elem['inlier1'], elem['inlier2'])
            point3D = -np.array((point4D[0]/point4D[3], point4D[1]/point4D[3], point4D[2]/point4D[3]))
            # Check if z coordinate is below 0, which indicates an incorrect camera pose
            if point3D[2] < 0:
                return None
            else:
                points3D.append(point3D)
        return np.array(points3D)
    
    # Calculate all possible candidates
    U, _, Vt = np.linalg.svd(np.matrix(E_matrix))
    W = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    R1 = U @ W @ Vt
    R2 = U @ np.transpose(W) @ Vt
    t = U[:, 2]
    
    P_list = []
    for R in (R1, R2):
        P_list.append(intrinsic_K @ np.concatenate((R, t), axis=1))
        P_list.append(intrinsic_K @ np.concatenate((R, -t), axis=1))
    P_list = np.array(P_list)

    # Check which of the 4 candidates is the correct one
    for P in P_list:
        points3D = triangulation_and_pose_check(P, epi_geo)
        if points3D is not None:
            # Write P matrix to file
            if viz == True:
                filename = "camera_pose"
                utility.write_data_to_file(RES_DIR, filename, P, extension='numpy')
            return (P, points3D.reshape((points3D.shape[0], 3)))
    else:
        raise Exception("None of the 4 possible candidate is valid!")


def task_5_obtain_point_cloud(points3D, viz=True):
    # Create point cloud and populate it with 3D points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points3D)
    
    # Save point cloud in file
    if viz == True:
        filename = "point_cloud"
        utility.write_data_to_file(RES_DIR, filename, point_cloud, extension='ply')
    
    return point_cloud


if __name__ == '__main__':
    program = CommandLine()
    program.parse_args()
    images = utility.load_images_from_dir(IMG_DIR)
    intrinsic_params = utility.load_intrinsic_params_from_txt(IMG_DIR, INTRINSIC_FILE)
    for image_pair in permutations(images, 2):
        features = task_1_feature_extraction(image_pair)
        (F_matrix, epi_geo) = task_2_estimate_fundamental_matrix(image_pair, features)
        E_matrix = task_3_compute_essential_matrix(F_matrix, intrinsic_params)
        P_matrix, points3D = task_4_obtain_camera_pose(E_matrix, intrinsic_params, epi_geo)
        point_cloud = task_5_obtain_point_cloud(points3D)

