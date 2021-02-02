
%% Demo.m 
% All the codes in "code_forMetrics" are from MIT Saliency Benchmark (https://github.com/cvzoya/saliency). Please refer to their webpage for more details.

% load global parameters, you should set up the "ROOT_DIR" to your own path
% for data.
clear all
% METRIC_DIR = 'code_forMetrics';
% addpath(genpath(METRIC_DIR));

%% path to store evaluation results
CACHE = ['cache/'];
if ~exist(CACHE, 'dir')
    mkdir(CACHE);
end
%%
% options.Result_path = '/home/qx/Downloads/DHF1Kres1015/imgs/';
% options.Result_path = '/home/qx/Downloads/';
options.Result_path = 'R:/dept2/qxlai/DataSets/';
% options.Result_path = '/media/qx/Seagate Expansion Drive/tmp/';
% options.Result_path = '/home/qx/Downloads/UNISAL/';
% options.DS_path = '/home/qx/DataSets/';
options.DS_path = 'R:/dept2/qxlai/DataSets/';

Datasets{1} = 'UCF sports';
Datasets{2} = 'Hollywood-2';
Datasets{3} = 'DHF1K';


Metrics{1} = 'NSS'; 
Metrics{2} = 'similarity'; 
Metrics{3} = 'CC';
Metrics{4} = 'AUC_Judd';
Metrics{5} = 'AUC_shuffled';

Metrics_name{1} = 'Normalized Scanpath Saliency metric: ';
Metrics_name{2} = 'Similarity metric: ';
Metrics_name{3} = 'Cross-correlation metric: ';
Metrics_name{4} = 'AUC (Judd) metric: ';
Metrics_name{5} = 'shuffled AUC metric: ';

%Results{1} = 'MotionAwareTwoStream';
%Results{1} = 'UNISAL';
% Results{1} = 'SALLSTM';
% Results{1} = 'DHF1Ktest_chen_iv';
% Results{1} = 'results_final';
% Results{1} = 'DHF1K_fcsb5361_data012_loss01-12';
% Results{2} = 'DHF1K_fcsb5341_data2_loss04-25';
% Results{3} = 'DHF1K_fcsb5341_data2_loss04-20';
% Results{1} = 'sweet';
% Results{2} = 'snow';
% Results{1} = 'DHF1K_chen_191204';
% Results{1} = 'DHF1Ktest';

Results{1} = 'SalCLSTMN50_m2_20';
Results{1} = 'salmodg';
Results{1} = 'model_saliency';
Results{1} = 'salds2';
Results{1} = 'zhang-dhf1k_2';
% Results{1} = '3D_Unet_with_RCL';
Results{1} = 'ATN_v3';

% Results{1} = 'UNISAL';
% Results{2} = 'UNISAL_UCFSports-only';
% Results{3} = 'UNISAL_SALCION-only';
% Results{4} = 'UNISAL_Hollywood2-only';
% Results{5} = 'UNISAL_DHF1K-only';
% Results{6} = 'UNISAL_DHF1K-Hollywood2-UCFSports-only';

% Results{1} = 'DHF1K_fcsb5361_data012_lr4-5_data012-24';
% Results{2} = 'DHF1K_fcsb5361_data012_lr4-5_data012-24_data012-5';
% Results{3} = 'DHF1K_fcsb5361_data012_lr4-10_data012-24';
% Results{4} = 'DHF1K_fcsb5361_data012_lr4-34';

% fprintf('Thanks for submitting your saliency maps to the benchmark.\n');
for i = 3:length(Datasets)
    % disp(Datasets{i});
    for k =1:1 %1: length(Results)
        % saliency prediction results
        % options.SALIENCY_DIR = [options.Result_path Datasets{i} '/' Results{k} '/'];
        options.SALIENCY_DIR = [options.Result_path Results{k} '/'];
        % disp(Results{k});
        % fprintf('--------------------------------------------------------------\n');
        %fprintf('The scores of %s are: \n\n', Results{k});
        fprintf('%s\n', Results{k});
        % dataset path
        if isequal(Datasets{i}, 'DHF1K'),
            options.DS_GT_DIR = [options.DS_path Datasets{i} '/eval/'];
        else
            options.DS_GT_DIR = [options.DS_path Datasets{i} '/test/'];
        end
        options.IMG_DIR = [options.DS_GT_DIR, '*/images/'];
                
        for j =1:length(Metrics)
            if ~exist([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'file')                 
                [result, allMetric, ~] = evaluationFunc(options, Metrics{j});
                save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
                % save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '_all.mat'], 'result');
                % std_value = std(allMetric); % calculate std value if you want to
            else
                load([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat']);
            end
            meanMetric{i}(k,j) = result;
            fprintf('%s :%.4f \n', Metrics{j}, result);
            %fprintf('%s%.4f \n', Metrics_name{j}, result);
        end
        fprintf('\n');
    end
end
% fprintf('Regards,\nDHF1K Benchmark Team\n');
%%
