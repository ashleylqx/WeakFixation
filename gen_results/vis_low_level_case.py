
import os
import cv2
import random
import numpy as np
import torch
import sal_metrics

DataSet_path = '/Users/qiuxia/Downloads/DataSets'
Result_path = '/Users/qiuxia/Downloads/WF_Preds'
Save_path = '/Users/qiuxia/Downloads/WF_Preds/low_level_case/'
Examples = {'MIT1003':['syntheticData1.jpeg', 'syntheticData2.jpeg', 'syntheticData3.jpeg', 'syntheticData4.jpeg'],
            'toronto':['72.jpg', '111.jpg']}
Methods = ['ours', 'BMS', 'gbvs', 'EML-NET', 'UNISAL']

"""                      
Availble font lists
FONT_HERSHEY_COMPLEX
FONT_HERSHEY_COMPLEX_SMALL
FONT_HERSHEY_DUPLEX
FONT_HERSHEY_PLAIN
FONT_HERSHEY_SCRIPT_COMPLEX
FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_HERSHEY_SIMPLEX
FONT_HERSHEY_TRIPLEX
FONT_ITALIC
"""
fontscale = 1.0
# (B, G, R)
color = (0, 0, 255)
# select font
fontface = cv2.FONT_HERSHEY_COMPLEX

for dataset in Examples.keys():
    dataset_dir = os.path.join(DataSet_path, dataset)
    if dataset == 'MIT1003':
        path_saliency = os.path.join(dataset_dir, 'ALLFIXATIONMAPS')
        path_fixation = os.path.join(dataset_dir, 'ALLFIXATIONS')
        path_image = os.path.join(dataset_dir, 'ALLSTIMULI')
    else:
        path_saliency = os.path.join(dataset_dir, 'maps')
        path_fixation = os.path.join(dataset_dir, 'fixation')
        path_image = os.path.join(dataset_dir, 'images')

    pred_files = Examples[dataset]
    for method in Methods:
        out_folder = os.path.join(Result_path, dataset, method)
        # pred_files = os.listdir(out_folder)

        for file_name in pred_files:
            if dataset == 'MIT1003':
                sal_path = os.path.join(path_saliency, file_name.replace('.jpeg', '_fixMap.jpg'))
                fix_path = os.path.join(path_fixation, file_name.replace('.jpeg', '_fixPts.jpg'))
                result_name = file_name.replace('.jpeg', '.png')

            else:
                sal_path = os.path.join(path_saliency, file_name.replace('.jpg', '.png'))
                fix_path = os.path.join(path_fixation, file_name.replace('.jpg', '.png'))
                result_name = file_name.replace('.jpg', '.png')

            img = cv2.imread(os.path.join(path_image, file_name))
            height, width, _ = img.shape

            save_name = '%s_%s_%s' % (dataset, method, result_name)
            save_name_gt = '%s_%s_%s' % (dataset, 'gt', result_name)

            # ground truth
            if not os.path.exists(os.path.join(Save_path, save_name_gt)):
                sal_map_np = cv2.imread(sal_path, 0)
                # sal_map_np = sal_map_np[np.newaxis, :].astype('float')
                # fix_map_np = cv2.imread(fix_path, 0)
                # fix_map_np = fix_map_np > 0
                # fix_map_np = fix_map_np[np.newaxis, :].astype('uint8')

                ##sal_map_np = cv2.GaussianBlur(sal_map_np, (7, 7), 0)
                heatmap = cv2.applyColorMap(cv2.resize(sal_map_np, (width, height)), cv2.COLORMAP_JET)
                result = heatmap * 0.3 + img * 0.5
                # cv2.imshow("gt", result)
                # cv2.waitKey()
                cv2.imwrite(os.path.join(Save_path, save_name_gt), result)

            # other methods
            pred_final_np = cv2.imread(os.path.join(out_folder, result_name), 0)
            # pred_final_np = pred_final_np[np.newaxis, :].astype('float')
            heatmap = cv2.applyColorMap(cv2.resize(pred_final_np, (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5

            # text_img = np.zeros((height//4, width, 3), np.uint8)
            # text_img.fill(255)
            # cv2.putText(text_img, "AUC-J 0.88 NSS 1.02", (25, 40), fontface, fontscale, color)
            # cv2.imwrite(os.path.join(Save_path, save_name), np.vstack([result, text_img]))

            # # cv2.putText(result, "FONT_HERSHEY_COMPLEX", (25, 40), fontface, fontscale, color)
            #
            # # cv2.imshow("method", result)
            # # cv2.waitKey()
            cv2.imwrite(os.path.join(Save_path, save_name), result)
