
import os
import cv2
import random
import numpy as np
import torch
import sal_metrics

# TODO: pending test for aucs
def checkBounds(dim, data):
    pts = np.round(data)
    valid = np.sum((pts < np.tile(dim, [pts.shape[0], 1])), 1) # pts < image dimensions
    valid = valid + np.sum((pts >= 0), 1)  #pts > 0
    data = data[valid == 4, :]
    return data

def makeFixationMap(dim,pts):
    # pdb.set_trace()
    pts = np.round(pts)
    map = np.zeros(dim)
    pts = checkBounds(dim, pts)
    # pdb.set_trace()
    pts = pts.astype('int')
    map[(pts[:,0], pts[:,1])] += 1

    return map

def other_maps(results_size, path_fixation, pred_files, n_aucs_maps=10):
    """Sample reference maps for s-AUC"""
    # while True:
        # this_map = np.zeros(results_size[-2:])
    ids = random.sample(range(len(pred_files)), min(len(pred_files), n_aucs_maps))
    # pdb.set_trace()
    for k in range(len(ids)):
        fix_path = os.path.join(path_fixation, pred_files[ids[k]][:-4] + '_fixPts.jpg')
        fix_map_np = cv2.imread(fix_path, 0)
        fix_map_np = (fix_map_np > 0).astype('float')
        training_resolution = fix_map_np.shape
        # pdb.set_trace()
        rescale = np.array(results_size)/np.array(training_resolution)
        rows, cols = np.where(fix_map_np)
        pts = np.vstack([rows, cols]).transpose()

        if 'fixation_point' not in locals():
            fixation_point = pts.copy()*np.tile(rescale, [pts.shape[0], 1])
        else:
            fixation_point = np.vstack([fixation_point, pts*np.tile(rescale, [pts.shape[0], 1])])

    other_map = makeFixationMap(results_size, fixation_point)
    pdb.set_trace()
    return other_map
        # yield other_map

DataSet_path = '/Users/qiuxia/Downloads/DataSets'
Result_path = '/Users/qiuxia/Downloads/WF_Preds'
Examples = {'MIT1003':['syntheticData1.jpeg', 'syntheticData2.jpeg', 'syntheticData3.jpeg', 'syntheticData4.jpeg'],
            'toronto':['72.jpg', '111.jpg']}
Methods = ['ours', 'BMS', 'gbvs', 'EML-NET', 'UNISAL']
# ('kld', 'nss', 'cc', 'sim', 'aucj', 'aucs')
metrics = ('nss', 'cc', 'sim', 'aucj')

record_file = os.path.join(Result_path, 'eval_low_level_case.txt')
eval_lines = []


for dataset in Examples.keys():
    dataset_dir = os.path.join(DataSet_path, dataset)
    if dataset == 'MIT1003':
        path_saliency = os.path.join(dataset_dir, 'ALLFIXATIONMAPS')
        path_fixation = os.path.join(dataset_dir, 'ALLFIXATIONS')
    else:
        path_saliency = os.path.join(dataset_dir, 'maps')
        path_fixation = os.path.join(dataset_dir, 'fixation')

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

            sal_map_np = cv2.imread(sal_path, 0)
            fix_map_np = cv2.imread(fix_path, 0)
            fix_map_np = fix_map_np > 0
            sal_map_np = sal_map_np[np.newaxis, :].astype('float')
            fix_map_np = fix_map_np[np.newaxis, :].astype('uint8')
            sal_map = torch.tensor(sal_map_np, dtype=torch.float)
            fix_map = torch.tensor(fix_map_np, dtype=torch.uint8)

            pred_final_np = cv2.imread(os.path.join(out_folder, result_name), 0)
            pred_final_np = pred_final_np[np.newaxis, :].astype('float')
            pred_final = torch.tensor(pred_final_np, dtype=torch.float)

            content = '%s\t%s\t%s' % (dataset, method, file_name)

            for this_metric in metrics:
                if this_metric == 'sim':
                    # sim_val = sal_metrics.similarity(pred_final, sal_map)
                    sim_val = sal_metrics.similarity(pred_final_np, sal_map_np)  # ok!
                    # eval_lines.append('%s\t%s\t%s\t%.4f\n' % (dataset, method, this_metric, sim_val))
                    content = content + '\t%s\t%.4f' % (this_metric, sim_val)

                elif this_metric == 'aucj':
                    aucj_val = sal_metrics.auc_judd(pred_final_np, fix_map_np)  # ok!
                    # eval_lines.append('%s\t%s\t%s\t%.4f\n' % (dataset, method, this_metric, aucj_val))
                    content = content + '\t%s\t%.4f' % (this_metric, aucj_val)

                elif this_metric == 'aucs':
                    other_map = other_maps(pred_final_np.shape[-2:], path_fixation, pred_files, n_aucs_maps=10)
                    # pdb.set_trace()
                    aucs_val = sal_metrics.auc_shuff_acl(pred_final_np, fix_map_np, other_map)  # 0.715 not equal to 0.74
                    # eval_lines.append('%s\t%s\t%s\t%.4f\n' % (dataset, method, this_metric, aucs_val))
                    content = content + '\t%s\t%.4f' % (this_metric, aucs_val)

                elif this_metric == 'kld':
                    kld_val = sal_metrics.kld_loss(pred_final, sal_map)
                    # eval_lines.append('%s\t%s\t%s\t%.4f\n' % (dataset, method, this_metric, kld_val))
                    content = content + '\t%s\t%.4f' % (this_metric, kld_val)

                elif this_metric == 'nss':
                    nss_val = sal_metrics.nss(pred_final, fix_map)  # do not need .exp() for our case; ok!
                    # eval_lines.append('%s\t%s\t%s\t%.4f\n' % (dataset, method, this_metric, nss_val))
                    content = content + '\t%s\t%.4f' % (this_metric, nss_val)

                elif this_metric == 'cc':
                    cc_val = sal_metrics.corr_coeff(pred_final, sal_map)  # do not need .exp() for our case; ok!
                    # eval_lines.append('%s\t%s\t%s\t%.4f\n' % (dataset, method, this_metric, cc_val))
                    content = content + '\t%s\t%.4f' % (this_metric, cc_val)

            content = content + '\n'
            eval_lines.append(content)

with open(record_file, 'w') as f:
    f.writelines(eval_lines)