import cv2 as cv
import numpy as np
import argparse
import math
import re
from tabulate import tabulate
import os
from tools import *
from skimage.measure import ransac
from skimage.measure import LineModelND
from skimage.transform import ProjectiveTransform, AffineTransform
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Fresco Reconstruction.')
    parser.add_argument('--fragments_dir_path', help='Path to input the fragments directory path.', default='src/images/frag_eroded/')
    parser.add_argument('--fresco_path', help='Path to input fresco image.', default='src/images/image.jpg')
    parser.add_argument('--all', help='Do we compute all the fragments', default=True)
    parser.add_argument('--index_fragment', help='The index of the fragment to compute', default=92)
    return parser.parse_args()    

def recup_fragments_path(fragements_dir_path):
    total = 0
    for root, dirs, files in os.walk(fragements_dir_path):
        for file in files:
            n = int((file.split('_')[2]).split('.')[0])
            if n > total: total = n

    fragments_path = ['frag_eroded_'+str(i)+'.png' for i in range(0, total+1)]
    
    return fragments_path
    
def detect_interest_point_using_SIFT(img_fragment, img_fresco):
    #-- Detect keypoint using SIFT detector 
    detector = cv.SIFT.create()
    keypoints_fragment, descriptors_fragment = detector.detectAndCompute(img_fragment, None)
    keypoints_fresco, descriptors_fresco = detector.detectAndCompute(img_fresco, None)
    
    return keypoints_fragment, descriptors_fragment, keypoints_fresco, descriptors_fresco
    
def match_interest_point_using_SIFT(descriptors_fragment, descriptors_fresco):
    #-- Match descriptor vectors
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    all_matches = matcher.knnMatch(descriptors_fragment, descriptors_fresco, 2)
        
    return all_matches
    
def filter_matches_using_Lowe_ratio_test(all_matches):
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.55
    good_matches = []
    for m,n in all_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            
    return good_matches

def filter_matches_using_RANSAC(keypoints_fragment, keypoints_fresco, all_matches):
    matches = []
    for m,n in all_matches: matches.append(m)
    
    src_pts = np.float32([ keypoints_fragment[m.queryIdx].pt for m in matches ]).reshape(-1, 2)
    dst_pts = np.float32([ keypoints_fresco[m.trainIdx].pt for m in matches ]).reshape(-1, 2)
    
    rng = np.random.default_rng()

    # generate coordinates of line
    x = np.arange(-200, 200)
    y = 0.2 * x + 20
    data = np.column_stack([src_pts, dst_pts])

    # add gaussian noise to coordinates
    noise = rng.normal(size=data.shape)
    data += 0.5 * noise
    data[::2] += 5 * noise[::2]
    data[::4] += 20 * noise[::4]

    # fit line using all data
    model = LineModelND()
    model.estimate(data)

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                                residual_threshold=8, max_trials=10000)
    outliers = inliers == False

    # generate coordinates of estimated models
    line_x = np.arange(-500, 500)
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_x)

    fig, ax = plt.subplots()
    ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
            label='Inlier data')
    ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
            label='Outlier data')
    ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
    ax.legend(loc='lower left')
    plt.show()
    
    n_inliers = np.sum(inliers)

    inlier_keypoints_left = [cv.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    good_matches = [cv.DMatch(idx, idx, 1) for idx in range(n_inliers)]
    
    src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in good_matches ]).reshape(-1, 2)
    dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in good_matches ]).reshape(-1, 2)
    
    return good_matches
    

def draw_matches(img_fragment, img_fresco, keypoints_fragment, keypoints_fresco, matches):
    #-- Draw matches and display it
    img_matches = np.empty((max(img_fragment.shape[0], img_fresco.shape[0]), img_fragment.shape[1]+img_fresco.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_fragment, keypoints_fragment, img_fresco, keypoints_fresco, matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches

def localize_fragment(keypoints_fragment, keypoints_fresco, good_matches):
    #-- Localize the object
    fragment_matches = np.empty((len(good_matches),2), dtype=np.float32)
    fresco_matches = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        fragment_matches[i,0] = keypoints_fragment[good_matches[i].queryIdx].pt[0]
        fragment_matches[i,1] = keypoints_fragment[good_matches[i].queryIdx].pt[1]
        fresco_matches[i,0] = keypoints_fresco[good_matches[i].trainIdx].pt[0]
        fresco_matches[i,1] = keypoints_fresco[good_matches[i].trainIdx].pt[1]
        
    # print(fragment_matches, fresco_matches)
    return fragment_matches, fresco_matches

def compute_fragment_position_coordonates(fragment_matches, fresco_matches):
    #-- Calcul of the position x by the average of the x fresco matches coordonates 
    x = 0
    for i in range(len(fresco_matches)):
        x = x + fresco_matches[i,0]
    x = x / len(fresco_matches)
    
    #-- Calcul of the position y by the average of the x fresco matches coordonates
    y = 0
    for i in range(len(fresco_matches)):
        y = y + fresco_matches[i,1]
    y = y / len(fresco_matches)
    
    #-- Compute of the rotation angle théta θ by the average of the x fresco matches coordonates
    v1 = np.array((fresco_matches[0][0]-fresco_matches[1][0], fresco_matches[0][1]-fresco_matches[1][1]))
    v2 = np.array((fragment_matches[0][0]-fragment_matches[1][0], fragment_matches[0][1]-fragment_matches[1][1]))
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    theta = - np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)) * 180 / np.pi
    if theta == np.nan or theta == np.NaN or theta == np.NAN :
        theta = 0 
    
    return x, y, theta

def detect_fragment_position(img_fragment, img_fresco, only_one_fragment): 
     #-- Detection and description of all interest point between the fragment and the fresco
    keypoints_fragment, descriptors_fragment, keypoints_fresco, descriptors_fresco = detect_interest_point_using_SIFT(img_fragment, img_fresco)
    
    #-- Match of all interest point between the fragment and the fresco
    all_matches = match_interest_point_using_SIFT(descriptors_fragment, descriptors_fresco)
    
    #-- Draw the all matches points
    if only_one_fragment:
        matches = []
        for m,n in all_matches: matches.append(m)
        img_fresco_with_allmatches = draw_matches(img_fragment, img_fresco, keypoints_fragment, keypoints_fresco, matches)
        cv.imshow("All matches", img_fresco_with_allmatches)
    
    #-- Filter the good matches
    good_matches = filter_matches_using_Lowe_ratio_test(all_matches)
    # good_matches = filter_matches_using_RANSAC(keypoints_fragment, keypoints_fresco, all_matches)
    # good_matches = filter_matches_using_Euclidian_Distance(all_matches)
    
    #-- Draw the good matches points gave by the Lowe ratio test
    if only_one_fragment:
        img_fresco_with_goodmatches = draw_matches(img_fragment, img_fresco, keypoints_fragment, keypoints_fresco, good_matches)
        cv.imshow("Only good matches", img_fresco_with_goodmatches)
    
    #-- Localize the fragment 
    fragment_matches, fresco_matches = localize_fragment(keypoints_fragment, keypoints_fresco, good_matches)
    
    #-- Compute the fragment position coordonates
    x, y, theta = compute_fragment_position_coordonates(fragment_matches, fresco_matches)
    
    return round(x), round(y), round(theta,3)


#-- Parse the arguments
args = parse_args()
fragments_path_list = recup_fragments_path(args.fragments_dir_path)

print("============= TI - Fresco Reconstruction =============")
print(" APP5 Info - Polytech Paris-Saclay ©")
print(" Julien SAVE & Alexis DA COSTA\n")
print(" Fresco image path to compute :", UNDER, args.fresco_path, RESET)
print(" Fragment directory path to compute :", UNDER, args.fragments_dir_path, RESET, "\n")


img_fresco = cv.imread(cv.samples.findFile(args.fresco_path))
if img_fresco is None:
    print('Could not open or find the fresco images!')
    exit(0)

if args.all == True:
    for fragment in fragments_path_list: 
        
        #-- Recovers the images and transforms them into matrices
        img_fragment = cv.imread(cv.samples.findFile(args.fragments_dir_path + fragment))
        fragment_index = re.findall("[\s\d]+", fragment)[0]
        if img_fragment is None:
            print('Could not open or find the fragment images!')
            exit(0)
        
        try:
            x, y, theta = detect_fragment_position(img_fragment, img_fresco, False)
            print("",fragment_index, x, y, theta)
        except Exception as e:
            pass
else : 
    #-- Recovers the images and transforms them into matrices
    img_fragment = cv.imread(cv.samples.findFile(args.fragments_dir_path + 'frag_eroded_'+ str(args.index_fragment)+ '.png'))
    fragment_index = args.index_fragment
    
    if img_fragment is None:
        print('Could not open or find the fragment images!')
        exit(0)
        
    print(" The fragment", fragment_index, "is at the coordonates : ")
    
    try:
        x, y, theta = detect_fragment_position(img_fragment, img_fresco, True)
        print("",fragment_index, x, y, theta)
    except Exception as e:
        print(fragment_index, "-> impossible", e)

    cv.waitKey(0)

print("======================================================")
