function [] = main()
    task = input('Press 1 for Image difference\nPress 2 for optical flow\n');
    if task == 1
        choice = input('Press 1 for taking pictures with the camera\nPress 2 for arm moving images\nPlace 3 for flower images\n');
        if choice == 1
           imagecapture();
           directory = '.\camera\';
        elseif choice == 2
           disp('Loading arm movement images');
           directory = '.\arm\';
        else
           disp('Loading flower images');
           directory = '.\flower\';
        end


        S = dir(fullfile(directory,'*.png')); 
        F = fullfile(directory,S(1).name);
        first_frame = imread(F);
        figure;
        for k = 2:numel(S)
            F = fullfile(directory,S(k).name);
            current_frame = imread(F);
            disp(['Calculating difference for frame',' ', num2str(k-1),' ','and frame',' ', num2str(k)])
            [original_difference] = image_difference(first_frame, current_frame);

            subplot(2,2,1);
            imshow(first_frame);
            subplot(2,2,2);
            imshow(current_frame);
            subplot(2,2,3);    
            imshow(uint8(original_difference), 'DisplayRange', []);
            pause(.5)
            first_frame = current_frame;
        end
        close all
    else
        
        blocksize = input('Provide blocksize\n');
        choice = input('Press 1 for taking pictures with the camera\nPress 2 for arm moving images\nPlace 3 for flower images\n');
        if choice == 1
            livecam(blocksize);
        elseif choice == 2
           disp('Loading arm movement images');
           directory = '.\arm\';
        else
           disp('Loading flower images');
           directory = '.\flower\';
        end
        S = dir(fullfile(directory,'*.png')); 
        F = fullfile(directory,S(1).name);
        first_frame = imread(F);
        M(1) = getframe;
        for k = 2:numel(S)
            F = fullfile(directory,S(k).name);
            current_frame = imread(F);
            figure;
            opticalflownew(first_frame, current_frame, blocksize);
            first_frame=current_frame;
            M(k) = getframe;
        end
        movie(M)
        save myMovie M;
        close all
    end

end