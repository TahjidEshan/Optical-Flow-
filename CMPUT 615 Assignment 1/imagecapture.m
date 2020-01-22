function [] = imagecapture()
    %images = zeros(numberofframes);
    success = mexMTF2('init','pipeline v img_source f');
    numberofframes = input('How many frames should the camera capture?');
    interval = input('How many seconds should the system wait between each frame?');
    for i=1:numberofframes
        [~, image] = mexMTF2('get_frame');
        imwrite(image, strcat('/cshome/tahjid/Optical-Flow-/CMPUT 615 Assignment 1/camera/',num2str(i,'%d'),'.png'));
        pause(interval);
    end
end
    