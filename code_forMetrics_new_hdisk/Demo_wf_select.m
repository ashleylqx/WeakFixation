
%% Demo.m 
% All the codes in "code_forMetrics" are from MIT Saliency Benchmark (https://github.com/cvzoya/saliency). Please refer to their webpage for more details.

% load global parameters, you should set up the "ROOT_DIR" to your own path
% for data.
clear all
% METRIC_DIR = 'code_forMetrics';
% addpath(genpath(METRIC_DIR));

%% path to store evaluation results
CACHE = ['cache_wf/'];
if ~exist(CACHE, 'dir')
    mkdir(CACHE);
end
%%
% options.Result_path = '/home/qx/WF/Preds/';
% options.DS_path = '/home/qx/DataSets/';
% base_path = '/media/qx/dgx-r69-1/';
% base_path = '/home/qx/';
base_path = 'R:/dept2/qxlai/';

%  img_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image/';

% Datasets{1} = 'MIT1003';
Datasets{1} = 'SALICON';
% Datasets{1} = 'PASCAL-S';
% Datasets{2} = 'Hollywood-2';
% Datasets{3} = 'DHF1K';


% options.Result_path = [base_path, 'WF/Preds/', DataSets{i}, '/'];
options.DS_path = [base_path, 'DataSets/'];


Metrics{1} = 'NSS'; 
Metrics{2} = 'similarity'; 
Metrics{3} = 'CC';
Metrics{4} = 'AUC_Judd';
Metrics{5} = 'AUC_shuffled';

Results{1} = 'pred_map';
Results{2} = 'object_mask';

% Results{1} = 'ours_0221';

%%
for i = 1:length(Datasets)
    disp(Datasets{i});
    
    if isequal(Datasets{i}, 'SALICON')
        img_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image/';
    else
        img_path = sprintf('H:/Codes/WF/Preds/%s/tmp_image/', Datasets{i});
    end

    options.Result_path = [base_path, 'WF/Preds/', Datasets{i}, '/'];
    
    % dataset path
    if isequal(Datasets{i}, 'MIT1003')
        options.DS_GT_DIR = [options.DS_path, Datasets{i} '/ALLFIXATIONMAPS/'];
        options.IMG_DIR = [options.DS_path, Datasets{i}, '/ALLSTIMULI/'];
        frames = dir(fullfile([options.IMG_DIR '*.jpeg']));
    elseif isequal(Datasets{i}, 'SALICON')
        options.DS_GT_DIR = [options.DS_path, Datasets{i} '/maps/train/'];
        options.IMG_DIR = [options.DS_path, Datasets{i}, '/images/train/']; %
        frames = dir(fullfile([options.IMG_DIR '*.jpg']));
    elseif isequal(Datasets{i}, 'PASCAL-S')
        options.DS_GT_DIR = [options.DS_path, Datasets{i} '/maps/'];
        options.IMG_DIR = [options.DS_path, Datasets{i}, '/images/']; 
        frames = dir(fullfile([options.IMG_DIR '*.jpg']));
    end
    
    
    for k = 1:length(Results)
%     for k =length(Results):-1:1    
        % saliency prediction results
        % options.SALIENCY_DIR = [options.Result_path Datasets{i} '/' Results{k} '/'];
        options.SALIENCY_DIR = [options.Result_path Results{k} '/'];
        disp(Results{k});
                   
        for j =1:1%length(Metrics)
            if ~exist([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'file') 
                if isequal(Datasets{i}, 'MIT1003')
                    [result, allMetric, ~] = evaluationFunc_wf(options, Metrics{j});
                elseif isequal(Datasets{i}, 'PASCAL-S')
                    [result, allMetric, ~] = evaluationFunc_wf_pascal(options, Metrics{j});
                elseif isequal(Datasets{i}, 'SALICON')
                    [result, allMetric, ~] = evaluationFunc_wf_salicon_train(options, Metrics{j});
                end
                save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
                save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '_all.mat'], 'allMetric');
                % std_value = std(allMetric); % calculate std value if you want to
            else
                load([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat']);
                load([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '_all.mat']);
%                 [result, allMetric, ~] = evaluationFunc_wf(options, Metrics{j});
%                 save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
            end
%             [result, allMetric, ~] = evaluationFunc_wf_salicon_train(options, Metrics{j});
%             save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'result');
%             save([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '_all.mat'], 'allMetric');
                
            meanMetric{i}(k,j) = result;
            fprintf('%s :%.4f \n', Metrics{j}, result);
            
        end
        
        %% save 20% images
        [~, pos] = sort(allMetric, 'descend');
        tmp_frames = frames(pos);
        if isequal(Datasets{1}, 'SALICON')
            for tidx=floor(length(allMetric)*0.2)+1:floor(length(allMetric)*0.4)
                copyfile([options.IMG_DIR, tmp_frames(tidx).name],...
                    [img_path, strrep(tmp_frames(tidx).name,'.jpg', sprintf('_%d.jpg', k))]);
            end
        else
            for tidx=1:floor(length(allMetric)*0.2)
                copyfile([options.IMG_DIR, tmp_frames(tidx).name],...
                    [img_path, tmp_frames(tidx).name]);
            end
        end
    end
end

%%
