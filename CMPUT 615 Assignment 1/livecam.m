function [] = livecam(blocksize)
%     success = mexMTF2('init','pipeline v img_source f')
    [~, first_frame] = mexMTF2('get_frame');
    while true
        [~, current_frame] = mexMTF2('get_frame');
        opticalflow(first_frame, current_frame, blocksize);
%         imshow(current_frame)
        first_frame = current_frame;
    end
end