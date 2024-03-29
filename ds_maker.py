import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from imutils import paths

dataset_name = 'data/ida/PRJ-3563_archive/PRJ-3563'
DIR = f'{dataset_name}/masks'
pts = list(paths.list_images(DIR))
pts_pre = []

for p in pts:
    if "_pre_" in p:
        pts_pre.append(p)

DSIZE = [224, 224]
MIN_SIZE = 15

iter = 0

for p in pts_pre:
    lbl_pre_p = p
    lbl_post_p = p.replace('_pre_', '_post_')
    im_pre_p = p.replace('masks', 'images')
    im_post_p = lbl_post_p.replace('masks', 'images')

    lbl_pre = cv2.imread(lbl_pre_p, -1)
    lbl_post = cv2.imread(lbl_post_p, -1)
    im_pre = cv2.imread(im_pre_p, -1)
    im_post = cv2.imread(im_post_p, -1)


    contours, hierarchy = cv2.findContours(lbl_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.minAreaRect(c)

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        if width < MIN_SIZE or height < MIN_SIZE:
            continue

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_lbl = cv2.warpPerspective(lbl_pre, M, (width, height))
        warped = cv2.warpPerspective(im_pre, M, (width, height))

        warped_lbl_post = cv2.warpPerspective(lbl_post, M, (width, height))
        warped_post = cv2.warpPerspective(im_post, M, (width, height))

        warped_lbl = cv2.resize(warped_lbl, DSIZE, interpolation=cv2.INTER_NEAREST)
        warped = cv2.resize(warped, DSIZE, interpolation=cv2.INTER_NEAREST)

        lbl_post_counts = list(np.unique(warped_lbl_post, return_counts=True))
        if 0 in lbl_post_counts[0]:
            lbl_post_counts[0], lbl_post_counts[1] = lbl_post_counts[0][1:], lbl_post_counts[1][1:]
        label = lbl_post_counts[0][np.argmax(lbl_post_counts[1])]

        warped_lbl_post = cv2.resize(warped_lbl_post, DSIZE, interpolation=cv2.INTER_NEAREST)

        warped_post = cv2.resize(warped_post, DSIZE, interpolation=cv2.INTER_NEAREST)

        '''cv2.imshow('dd', warped_lbl)
        cv2.imshow('ss', warped)
        cv2.imshow('dd2', warped_lbl_post*75)
        cv2.imshow('ss2', warped_post)
        cv2.waitKey(0)'''
        cv2.imwrite(f'DS/IDA-BD/i_A/{iter}.png', warped)
        cv2.imwrite(f'DS/IDA-BD/i_B/{iter}_{label}.png', warped_post)
        iter += 1

        if iter % 100 == 0:
            print(f'{iter} images extracted')

    '''plt.imshow(lbl_pre)
    plt.show()'''
#%%