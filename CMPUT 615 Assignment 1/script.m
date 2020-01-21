

D = 'C:\Users\Eshan\Documents\CMPUT 615 Assignment 1\armD32im1\';
S = dir(fullfile(D,'*.png')); % pattern to match filenames.
F = fullfile(D,S(1).name);
first_frame = imread(F);
imshow(first_frame)
for k = 2:numel(S)
    F = fullfile(D,S(k).name);
    current_frame = imread(F);
    imshow(current_frame)
    imshow(difference(first_frame, current_frame))
    S(k).data = I; % optional, save data.
end