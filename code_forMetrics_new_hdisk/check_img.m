%%

base_path = 'R:/dept2/qxlai/DataSets/3D_Unet_with_RCL';

folders = dir(base_path);
%%
for f=3:length(folders)
              imgs = dir([base_path,'/', folders(f).name, '/*.png']);
    
    for idx=1:length(imgs)
        img = imread([base_path,'/', folders(f).name, '/', imgs(idx).name]);
    end
end