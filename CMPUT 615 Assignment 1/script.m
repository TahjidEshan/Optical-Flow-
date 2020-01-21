

%D = 'C:\Users\Eshan\Documents\CMPUT 615 Assignment 1\Flower60im3\';
D = 'C:\Users\Eshan\Documents\CMPUT 615 Assignment 1\armD32im1\';

S = dir(fullfile(D,'*.png')); % pattern to match filenames.
F = fullfile(D,S(1).name);
first_frame = imread(F);
imshow(first_frame)
blocksize = 32
for k = 2:numel(S)
    F = fullfile(D,S(k).name);
    current_frame = imread(F);
    %imshow(current_frame)
    %calculating difference
    %difference = image_difference(first_frame, current_frame);
    %imshow(difference)
    
    %calculating optical flow
    %opticalflow(first_frame, current_frame, blocksize)
    opticalflowwithcorners(first_frame, current_frame, blocksize)
    
    %uncomment to detect changes between consecutive images
    first_frame = current_frame;
end

pause(.5)
close all