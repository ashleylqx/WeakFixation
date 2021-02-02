function fix_map = load_salicon_fix(fixation_path)
    I = load(fixation_path);
    fix_map = zeros(I.resolution(1), I.resolution(2));
    
    Pts = I.gaze;
    
    for i=1:length(Pts)
        points = Pts(i).fixations;
        
        for j=1:size(points,1)
            pt = points(j,:);
            fix_map(pt(2), pt(1)) = 1;
        end
    end   
    
end
    
            
        