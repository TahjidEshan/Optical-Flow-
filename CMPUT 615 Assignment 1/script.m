function [] = script(calculate_difference)
    % 
    % numofframes = 20;
    % imagecapture(numofframes,2);
    %D = 'C:\Users\Eshan\Documents\CMPUT 615 Assignment 1\Flower60im3\';
    D = '/cshome/tahjid/Optical-Flow-/CMPUT 615 Assignment 1/armD32im1/';
%     D = '/cshome/tahjid/Optical-Flow-/CMPUT 615 Assignment 1/camera/';
    S = dir(fullfile(D,'*.png')); % pattern to match filenames.
    F = fullfile(D,S(1).name);
    first_frame = imread(F);
    % imshow(first_frame);
    blocksize = 32;
    for k = 2:numel(S)
        F = fullfile(D,S(k).name);
        current_frame = imread(F);
        %imshow(current_frame)
        %calculating difference
        if calculate_difference
            disp(['Calculating difference for frame',' ', num2str(k-1),' ','and frame',' ', num2str(k)])
            [original_difference, normalized_difference] = image_difference(first_frame, current_frame);
            figure;
            subplot(2,2,1);
            imshow(first_frame);
            subplot(2,2,2);
            imshow(current_frame);
            subplot(2,2,3);    
            imshow(uint8(original_difference), 'DisplayRange', []);
%             subplot(2,2,4);
%             imshow(uint8(normalized_difference));
            
        end
        %calculating optical flow
%         figure;
%         hold on
        opticalflow(first_frame, current_frame, blocksize)
%         hold off
        %opticalflowwithcorners(first_frame, current_frame, blocksize)
%         pause(3)
        %uncomment to detect changes between consecutive images
        first_frame = current_frame;
        %close all
    end


    close all
end