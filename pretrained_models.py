#!/usr/bin/python

import os
import numpy as np

class super_glue(object):
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