function [] = imagecapture(numberofframes, timebetweenimages)
    %images = zeros(numberofframes);
    success = mexMTF2('init','pipeline v img_source f')
    for i=1:numberofframes
        [~, image] = mexMTF2('get_frame');
        imwrite(image, strcat('/cshome/tahjid/Optical-Flow-/CMPUT 615 Assignment 1/camera/',num2str(i,'%d'),'.png'))
        pause(timebetweenimages);
    end
end
    