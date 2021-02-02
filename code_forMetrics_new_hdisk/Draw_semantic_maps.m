%%

pred_path = 'H:/Codes/WF/Preds/SALICON_train/examples_0224/';
semantic_path = 'H:/Codes/WF/Preds/SALICON_train/semantic_map/';

%%
images = {'COCO_train2014_000000189472',...
          'COCO_train2014_000000215555',...
          'COCO_train2014_000000365817',...
          'COCO_train2014_000000399227',...
          'COCO_train2014_000000447773',...
          'COCO_train2014_000000467063',...
          'COCO_train2014_000000048595',...
          };
img_names = {{'_pred_62_chair.png','_pred_67_dining table.png',...
              '_pred_79_oven.png','_pred_82_refrigerator.png'},...
             {'_pred_44_bottle.png','_pred_47_cup.png',...
              '_pred_70_toilet.png','_pred_81_sink.png'},...
             {'_pred_01_person.png','_pred_18_dog.png',...
              '_pred_38_kite.png'},...
             {'_pred_01_person.png','_pred_28_umbrella.png',...
              '_pred_62_chair.png'},...
             {'_pred_01_person.png','_pred_27_backpack.png',...
              '_pred_35_skis.png'},...
             {'_pred_01_person.png','_pred_19_horse.png',...
              '_pred_21_cow.png'},...
             {'_pred_01_person.png','_pred_18_dog.png',...
              '_pred_41_skateboard.png'},...
             };
%%         
idx = 7 ;        
img_folder = images{idx};
map_path = [semantic_path, img_folder];
imgs = img_names{idx};

image = imread([pred_path, img_folder, '_1.jpg']);
figure;imshow(image);
% tmp_img = image;
%%
alpha = 0.4;
sigma = 3.0;
window=double(uint8(3*sigma*2)+1);  
H=fspecial('gaussian', window, sigma);

sigma2 = 15.0;
window2=double(uint8(3*sigma2*2)+1);  
H2=fspecial('gaussian', window2, sigma2);


%%
sem_map = {};
clear tmp_image
bg_map = imread([map_path, '/', img_folder, '_00_background.png', ]);
% bg_map = uint8(double(bg_map)/double(max(bg_map(:))) * 255);
% bg_map = imresize(bg_map, [size(image,1), size(image,2)]);
pre_maps = zeros(size(map,1),size(map,2));

for i=1:length(imgs)
    
    map = imread([map_path, '/', img_folder, imgs{i}]); 
    if i==1
        map = map-bg_map;
        map = map-mean(map(:));
    end
    map = imfilter(map,H,'replicate');
%     map = uint8(double(map)/double(max(map(:))) * 255);
    map = imresize(map, [size(image,1), size(image,2)]);
    half = zeros(size(map,1),size(map,2),3);
    sem_map{i} = map;
    
    if i==1
        pre_maps = map;
        tmp_map = uint8(double(map)/double(max(map(:))) * 255);
        half(:,:,1)=tmp_map; % red
%         clear tmp_image
    elseif i==2
        pre_maps = max(pre_maps, sem_map{i-1});
        tmp_map = zeros(size(map,1),size(map,2));    
        tmp_map(map>pre_maps)=map(map>pre_maps);
        tmp_map = imfilter(tmp_map,H2,'replicate');
        tmp_map = uint8(double(tmp_map)/double(max(tmp_map(:))) * 255);
        half(:,:,2)=tmp_map; % green

%         half(:,:,2)=map; % green
    elseif i==3
        pre_maps = max(pre_maps, sem_map{i-1});
        tmp_map = zeros(size(map,1),size(map,2));
        tmp_map(map>pre_maps)=map(map>pre_maps);
        tmp_map = imfilter(tmp_map,H2,'replicate');
        tmp_map = uint8(double(tmp_map)/double(max(tmp_map(:))) * 255);
        half(:,:,3)=tmp_map; % blue
        
        tmp_others = half(:,:,1);
        tmp_others(map>pre_maps)=tmp_others(map>pre_maps)*0.0;
        half(:,:,1) = tmp_others;
        tmp_others = half(:,:,2);
        tmp_others(map>pre_maps)=tmp_others(map>pre_maps)*0.0;
        half(:,:,2) = tmp_others;

%         half(:,:,3)=map; % blue
%         clear tmp_image
    elseif i==4
%         pre_maps = max(pre_maps, sem_map{i-1});
%         tmp_map = zeros(size(map,1),size(map,2));
%         tmp_map(map>pre_maps)=map(map>pre_maps);
%         tmp_map = imfilter(tmp_map,H,'replicate');
%         tmp_map = uint8(double(tmp_map)/double(max(tmp_map(:))) * 255);
%         half(:,:,1)=tmp_map;
%         half(:,:,2)=tmp_map; % yellow

        half(:,:,1)=map;
        half(:,:,2)=map; % yellow
    end
    if ~exist('tmp_image','var')
%         tmp_image =  alpha*image + (1-alpha)/length(imgs)*uint8(half);
        tmp_image =  alpha*image + (1-alpha)*uint8(half);
    else
%         tmp_image = tmp_image + (1-alpha)/length(imgs)*uint8(half);
        tmp_image = tmp_image + (1-alpha)*uint8(half);
    end
    
    figure;imshow(tmp_image);
    
end

%%
half = zeros(size(image,1), size(image,2),3);
sem_map = zeros(size(image,1), size(image,2),length(imgs));
sem_map_norm = zeros(size(image,1), size(image,2),length(imgs));
for i=1:length(imgs)
     map = imread([map_path, '/', img_folder, imgs{i}]); 
    if i==1
%         map = map-bg_map;
%         map = map-mean(map(:));
        map = map-bg_map*1.0;
        map = map-mean(map(:))*1.6; % for 'COCO_train2014_000000048595'
        map(:,38:end) = map(:,38:end)*0.0;
%         map = map-bg_map*8;
%         map = map-mean(map(:))*2.0; % for 'COCO_train2014_000000467063'
    elseif i==2
        map = imtranslate(map,[-12, -5],'FillValues',0); % for 'COCO_train2014_000000048595'
        map = map-mean(map(:))*16.0; % for 'COCO_train2014_000000048595'
        map(:,1:8) = map(:,1:8)*0.01;
    elseif i==3
        map = imtranslate(map,[-1.5, 5],'FillValues',0); % for 'COCO_train2014_000000048595'
        map(30:end,:) = map(30:end,:)*0.0;
    end
    map = imfilter(map,H,'replicate');
    map = imresize(map, [size(image,1), size(image,2)]);
    
    sem_map(:,:,i)=map;
    sem_map_norm(:,:,i)=uint8(double(map)/double(max(map(:))) * 255);
end

%% enhance certain channel, e.g. 2nd channel for 'COCO_train2014_000000447773'

sem_map(:,:,2)=sem_map(:,:,2)*1.0;
%%
sem_map(:,:,3)=sem_map(:,:,3)*0.3;

%%
ratios = [1, 0.1, 0.1, 0.1, 0.1;...
          0.1, 1, 0.1, 0.1, 0.1;...
          0.1, 0.1, 1, 0.1, 0.1;...
          0.1, 0.1, 0.1, 1, 1;...
          ];
for i=1:size(image,1)
    for j=1:size(image,2)
        [~, arg_max] = max(sem_map(i,j,:));
        ratio = ratios(arg_max,:);
        
        half(i,j,1) = sem_map_norm(i,j,1)*ratio(1);
        half(i,j,2) = sem_map_norm(i,j,2)*ratio(2);
        half(i,j,3) = sem_map_norm(i,j,3)*ratio(3);
        if size(sem_map,3)==4
            half(i,j,1) = half(i,j,1)+sem_map_norm(i,j,4)*ratio(4);
            half(i,j,2) = half(i,j,2)+sem_map_norm(i,j,4)*ratio(5);
        end   
    end
end

half = imfilter(half,H2,'replicate');
tmp_image =  alpha*image + (1-alpha)*uint8(half);
figure;imshow(tmp_image);

%%
imwrite(tmp_image, [semantic_path, img_folder, '_sm.jpg']);
%%
half = zeros(size(map,1),size(map,2),3);
for i=1:length(sem_map)
    
end

%%
images = dir([pred_path, '*.jpg']);

%%
for i=1:length(images)
    img_folder = images(i).name(1:end-6);
    
end