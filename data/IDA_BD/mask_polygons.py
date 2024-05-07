#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          # 
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, # 
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################

#####
# FORKED AND UPDATED FROM
# https://github.com/DIUx-xView/xView2_baseline/tree/master
#####

import json
from os import path, walk, makedirs
from sys import exit, stderr

from cv2 import fillPoly, imwrite
import numpy as np
from shapely import wkt
from shapely.geometry import mapping, Polygon
from skimage.io import imread
from tqdm import tqdm
import imantics 

# This removes the massive amount of scikit warnings of "low contrast images"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_dimensions(file_path):
    """
    :param file_path: The path of the file 
    :return: returns (width,height,channels)
    """
    # Open the image we are going to mask
    pil_img = imread(file_path)
    img = np.array(pil_img)
    w, h, c = img.shape
    return (w, h, c)


def mask_polygons_separately(size, shapes):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :returns: a dict of masked polygons with the shapes filled in from cv2.fillPoly
    """
    # For each WKT polygon, read the WKT format and fill the polygon as an image
    masked_polys = {}

    for u in shapes:
        sh = shapes[u]
        mask_img = np.zeros(size, np.uint8)
        i = fillPoly(mask_img, [sh], (255, 255, 255))
        masked_polys[u] = i

    return masked_polys

def mask_polygons_together(size, shapes, dmgs):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :param dmgs: dict of object damage state
    :returns: A numpy array with the polygons filled 255s where there's a building and 0 where not 
    """
    # For each WKT polygon, read the WKT format and fill the polygon as an image
    mask_img = np.zeros(size, np.uint8)

    for u in shapes:
        blank =  np.zeros(size, np.uint8)
        poly = shapes[u] 
        fillPoly(blank, [poly], (1, 1, 1))
        damage_state = dmgs[u]
        blank[blank > 1] = 0
        blank[blank == 1] = damage_state
        mask_img += blank

    return mask_img

def mask_polygons_together_with_border(size, shapes, dmgs, border):
    """
    :param size: A tuple of the (width,height,channels)
    :param shapes: A list of points in the polygon from get_feature_info
    :param dmgs: dict of object damage state
    :returns: a dict of masked polygons with the shapes filled in from cv2.fillPoly
    """

    # For each WKT polygon, read the WKT format and fill the polygon as an image
    mask_img = np.zeros(size, np.uint8)

    for u in shapes:
        blank =  np.zeros(size, np.uint8)
        # Each polygon stored in shapes is a np.ndarray
        poly = shapes[u]
        damage_state = dmgs[u]
        
        # Creating a shapely polygon object out of the numpy array 
        polygon = Polygon(poly)

        # Getting the center points from the polygon and the polygon points
        (poly_center_x, poly_center_y) = polygon.centroid.coords[0]
        polygon_points = polygon.exterior.coords

        # Setting a new polygon with each X,Y manipulated based off the center point
        shrunk_polygon = []
        for (x,y) in polygon_points:
            if x < poly_center_x:
                x += border
            elif x > poly_center_x:
                x -= border

            if y < poly_center_y:
                y += border
            elif y > poly_center_y:
                y -= border

            shrunk_polygon.append([x,y])
        
        # Transforming the polygon back to a np.ndarray
        ns_poly = np.array(shrunk_polygon, np.int32)
  
        # Filling the shrunken polygon to add a border between close polygons
        fillPoly(blank, [ns_poly], (1, 1, 1))
        blank[blank > 1] = 0
        blank[blank == 1] = damage_state
        mask_img += blank

    return mask_img

def save_masks(masks, output_path, mask_file_name):
    """
    :param masks: dictionary of UID:masked polygons from mask_polygons_separately()
    :param output_path: path to save the masks
    :param mask_file_name: the file name the masks should have 
    """
    # For each filled polygon, write out a separate file, increasing the name
    for m in masks:
        final_out = path.join(output_path,
                              mask_file_name + '_{}.png'.format(m))
        imwrite(final_out, masks[m])

def save_one_mask(masks, output_path, mask_file_name):
    """
    :param masks: list of masked polygons from the mask_polygons_separately function 
    :param output_path: path to save the masks
    :param mask_file_name: the file name the masks should have 
    """
    # For each filled polygon, write the mask shape out to the file per image
    mask_file_name = path.join(output_path, mask_file_name + '.png').replace('_pre', '_post')
    mask_file_pre = mask_file_name.replace('_post', '_pre')
    mask_post = masks[:, :, 0]
    mask_pre = np.where(mask_post == 0, 0, 255)
    imwrite(mask_file_name, mask_post)
    imwrite(mask_file_pre, mask_pre)
    

def read_json(json_path):
    """
    :param json_path: path to load json from
    :returns: a python dictionary of json features
    """
    annotations = json.load(open(json_path))
    return annotations


def get_feature_info(feature, feature_post):
    """
    :param feature: a python dictionary of json labels
    :param feature_post: json post_disaster labels
    :returns: a list mapping of polygons contained in the image 
    """
    # Getting each polygon points from the json file and adding it to a dictionary of uid:polygons
    props = {}

    post_dmg = {}
    for xy in feature_post['features']['xy']:
        post_dmg[xy['properties']['uid']] = d_map[xy['properties']['subtype']]

    for feat in feature['features']['xy']:
        feat_shape = wkt.loads(feat['wkt'])
        coords = list(mapping(feat_shape)['coordinates'][0])
        props[feat['properties']['uid']] = (np.array(coords, np.int32))

    return props, post_dmg


def mask_chips(json_path, images_directory, output_directory, single_file, border):
    """
    :param json_path: path to find multiple json files for the chips
    :param images_directory: path to the directory containing the images to be masked
    :param output_directory: path to the directory where masks are to be saved
    :param single_file: a boolean value to see if masks should be saved a single file or multiple
    """
    # For each feature in the json we will create a separate mask
    # Getting all files in the directory provided for jsons
    jsons = [j for j in next(walk(json_path))[2] if '_pre' in j]

    # After removing non-json items in dir (if any)
    for j in tqdm([j for j in jsons if j.endswith('json')],
                  unit='poly',
                  leave=False):
        # Our chips start off in life as PNGs
        j_post = j.replace('pre', 'post')
        chip_image_id = path.splitext(j)[0] + '.tif'
        mask_file = path.splitext(j)[0]

        # Loading the per chip json
        j_full_path = path.join(json_path, j)
        chip_json = read_json(j_full_path)
        chip_json_post = read_json(path.join(json_path, j_post))

        # Getting the full chip path, and loading the size dimensions
        chip_file = path.join(images_directory, chip_image_id)
        chip_size = get_dimensions(chip_file)

        # Reading in the polygons from the json file
        polys, dmgs = get_feature_info(chip_json, chip_json_post)

        # Getting a list of the polygons and saving masks as separate or single image files
        if len(polys) > 0:
            if single_file:
                if border > 0:
                    masked_polys = mask_polygons_together_with_border(chip_size, polys, dmgs, border)
                else:
                    masked_polys = mask_polygons_together(chip_size, polys, dmgs)
                save_one_mask(masked_polys, output_directory, mask_file)
            else:
                masked_polys = mask_polygons_separately(chip_size, polys)
                save_masks(masked_polys, output_directory, mask_file)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """mask_polygons.py: Takes in xBD dataset and masks polygons in the image\n\n
        WARNING: This could lead to hundreds of output images per input\n""")

    parser.add_argument('--input',
                        required=True,
                        metavar="/path/to/xBD/",
                        help='Path to parent dataset directory "xBD"')
    parser.add_argument('--single-file', 
                        action='store_true',
                        help='use to save all masked polygon instances to a single file rather than one polygon per mask file')
    parser.add_argument('--border',
                        default=0,
                        type=int,
                        metavar="positive integer for pixel border (e.g. 1)",
                        help='Positive integer used to shrink the polygon by')

    args = parser.parse_args()

    # disaster mapping
    global d_map
    d_map = {
        'no-damage': 1,
        'minor-damage': 2,
        'major-damage': 3,
        'destroyed': 4,
        'un-classified': 0
    }

    # Getting the list of the disaster types under the xBD directory
    disasters = next(walk(args.input))[1]

    for disaster in tqdm(disasters, desc='Masking', unit='disaster'):
        # Create the full path to the images, labels, and mask output directories
        image_dir = path.join(args.input, disaster, 'images')
        json_dir = path.join(args.input, disaster, 'labels')
        output_dir = path.join(args.input, disaster, 'masks')

        if not path.isdir(image_dir):
            print(
                "Error, could not find image files in {}.\n\n"
                .format(image_dir),
                file=stderr)
            exit(2)

        if not path.isdir(json_dir):
            print(
                "Error, could not find labels in {}.\n\n"
                .format(json_dir),
                file=stderr)
            exit(3)

        if not path.isdir(output_dir):
            makedirs(output_dir)

        mask_chips(json_dir, image_dir, output_dir, args.single_file, args.border)
