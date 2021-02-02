%% 1) select examples for ablation study
% modified from Demo_wf_select.m
clear all

%% path to store evaluation results
CACHE = ['cache_ab/'];
if ~exist(CACHE, 'dir')
    mkdir(CACHE);
end
%%
base_path = 'R:/dept2/qxlai/';
options.Result_path = 'H:/Codes/WF/Examples/backup/';

Datasets{1} = 'MIT1003';

options.DS_path = [base_path, 'DataSets/'];

Metrics{1} = 'NSS'; 
Metrics{2} = 'similarity'; 
Metrics{3} = 'CC';
Metrics{4} = 'AUC_Judd';
Metrics{5} = 'AUC_shuffled';

Results{1} = 'MIT1003_ours';   % full model
Results{2} = 'nopsal';          % no proposal; worse for large object? fail to highlight
Results{3} = 'noGrid';          % no grid; worse for small/no object? fail to highlight
Results{4} = 'noobj';           % no object; overall worse? fail to highlight

Results{5} = 'norn';            % no relation; relative saliency is not right
Results{6} = 'nomask';          % no competetion; worse for multi-object?

Results{7} = 'cbG16';           % with global center bias; emm, worse ...
Results{8} = 'nobs';            % no center bias; emm, worse ...

Results{9} = 'rf0.1_hth0.0';    % without info loss
Results{10} = 'rf0.0_hth0.1';    % without prior loss
Results{11} = 'rf0.0_hth0.0';   % with cls loss only

Results{12} = 'prior_gbvs';     % already have examples
Results{13} = 'prior_bms';      % already have examples
Results{14} = 'nips08';      % already have examples

%%
for i = 1:length(Datasets)
    disp(Datasets{i});
    
    if isequal(Datasets{i}, 'SALICON')
%         img_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image/';
        img_path = 'H:/Codes/WF/Preds/SALICON_train/tmp_image_ab/';
    else
%         img_path = sprintf('H:/Codes/WF/Preds/%s/tmp_image/', Datasets{i});
        img_path = sprintf('H:/Codes/WF/Preds/%s/tmp_image_ab/', Datasets{i});
    end

%     options.Result_path = [base_path, 'WF/Preds/', Datasets{i}, '/'];
    
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
    % I know that length(Metrics)==1
%     AM{i} = zeros(length(Results), length(frames));
    
    for k = 2:length(Results)
%     for k =length(Results):-1:1    
        % saliency prediction results
        % options.SALIENCY_DIR = [options.Result_path Datasets{i} '/' Results{k} '/'];
        options.SALIENCY_DIR = [options.Result_path Results{k} '/'];
        disp(Results{k});
                   
        for j =1:1%length(Metrics)
            if ~exist([CACHE Datasets{i} '_' Results{k} '_' Metrics{j} '.mat'], 'file') 
                if isequal(Datasets{i}, 'MIT1003')
                    [result, allMetric, ~] = evaluationFunc_wf_R2(options, Metrics{j});
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
%             AM{i}{k}{j} = allMetric;    
            AM{i}(k, :) = allMetric;
        end
        
        %% save 20% images
%         [~, pos] = sort(allMetric, 'descend');
%         tmp_frames = frames(pos);
%         if isequal(Datasets{1}, 'SALICON')
%             for tidx=floor(length(allMetric)*0.2)+1:floor(length(allMetric)*0.4)
%                 copyfile([options.IMG_DIR, tmp_frames(tidx).name],...
%                     [img_path, strrep(tmp_frames(tidx).name,'.jpg', sprintf('_%d.jpg', k))]);
%             end
%         else
%             for tidx=1:floor(length(allMetric)*0.2)
%                 copyfile([options.IMG_DIR, tmp_frames(tidx).name],...
%                     [img_path, tmp_frames(tidx).name]);
%             end
%         end
    end
end
%%
save([CACHE 'AB_metric_all.mat'], 'AM');
%% compare these results; exclude images
%             || cur_values(5)>cur_values(6) ...
%            || cur_values(7)<cur_values(8) ...
tmp_frames = frames;
cnt = 1;
NSS_values = AM{1};
while cnt<=size(NSS_values, 2)
    cur_values = NSS_values(:, cnt);
    [m, i] = max(cur_values);
    if any(cur_values==-1) || i~=1 || cur_values(4)~=min(cur_values(2:4)) ...
            || cur_values(11)~=min(cur_values(9:11))
        NSS_values(:, cnt)=[];
        tmp_frames(cnt) = [];
    else
        copyfile([options.IMG_DIR, tmp_frames(cnt).name],...
                    [img_path, tmp_frames(cnt).name]);
        cnt = cnt+1;
    end
end
%%
save([CACHE 'AB_selected_frames.mat'], 'tmp_frames');

%% 2) draw color visualization for ablation examples

images = {'i05june05_static_street_boston_p1010855',...
          'i1035544976',...
          'i1198772915',...
          'i2264606081',...
          'istatic_barcelona_street_city_outdoor_2_2005_img_0650',...
          };
postfix = {'_fixMap.jpg','_ours.png',...            
             '_nopsal.png','_noGrid.png','_noobj.png',...
             '_norn.png','_nomask.png',...
             '_nobs.png','_cbG16.png',...
             '_rf0.1_hth0.0.png',...
              '_rf0.0_hth0.1.png',...
             '_rf0.0_hth0.0.png',...
             '_prior_bms.png','_prior_gbvs.png','_nips08.png'...
             };

%%
alpha = 0.4;
sigma = 3.0;
window=double(uint8(3*sigma*2)+1);  
H=fspecial('gaussian', window, sigma);

sigma2 = 15.0;
window2=double(uint8(3*sigma2*2)+1);  
H2=fspecial('gaussian', window2, sigma2);

img_path = 'H:/Codes/WF/Preds/MIT1003/tmp_image_ab/';
%%
for i=1:length(images)
    img_name = images{i};
    image = imread([img_path img_name '.jpeg']);
    
    for j=length(postfix):length(postfix)  %3:length(postfix)
        map = imread([img_path img_name postfix{j}]);
        half = zeros(size(map,1),size(map,2),3);
        
        if j==1
%             alpha = 0.4;
            map = imfilter(map,H2,'replicate');
            tmp_map = uint8(double(map)/double(max(map(:))) * 255);
            half(:,:,1)=tmp_map; % red
            
        elseif j==2
%             alpha = 0.4;
            half(:,:,1)=map;
            half(:,:,2)=map; % yellow
            
        elseif j==length(postfix)
            map = double(map) - min(double(map(:)));
            tmp_map = uint8(double(map)/double(max(map(:))) * 255);
            half(:,:,2)=tmp_map; % green
        
        else
%             alpha = 0.4;
%             map = imfilter(map,H2,'replicate');
            map = double(map) - min(double(map(:)));
            tmp_map = uint8(double(map)/double(max(map(:))) * 255);
            half(:,:,3)=tmp_map; % blue
        end
        
        tmp_image =  alpha*image + (1-alpha)*uint8(half);
        figure;imshow(tmp_image);
        if j==1
            imwrite(tmp_image, [img_path img_name replace(postfix{j}, '.jpg', '_c.png')]);
        else
            imwrite(tmp_image, [img_path img_name replace(postfix{j}, '.png', '_c.png')]);
        end
    end
end



