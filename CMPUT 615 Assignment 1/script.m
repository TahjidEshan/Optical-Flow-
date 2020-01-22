function [] = script()

    D = '.\arm\';
;
    S = dir(fullfile(D,'*.png')); % pattern to match filenames.
    F = fullfile(D,S(1).name);
    first_frame = imread(F);
    % imshow(first_frame);
    blocksize = 4;
    for k = 2:numel(S)
        F = fullfile(D,S(k).name);
        current_frame = imread(F);
        disp("loaded new image")
        figure;
%         hold on
        disp("starting optical flow")
        opticalflownew(first_frame, current_frame, blocksize)
        disp("ending optical flow")
%         hold off
        %opticalflowwithcorners(first_frame, current_frame, blocksize)
%         pause(3)
        %uncomment to detect changes between consecutive images
%         first_frame = current_frame;
        %close all
    end


    close all
end