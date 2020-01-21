function [difference, bounding_box] = image_difference(frame1, frame2)
    [rows, columns, numberOfColorChannels] = size(frame1); 
    difference = zeros(rows, columns, numberOfColorChannels);
    threshold = 100;
    
    for i = 1:rows
        for j = 1:columns
            for k= 1:numberOfColorChannels
                if frame2(i,j,k)-frame1(i,j,k)>threshold
                    difference(i,j,k) = 1; 
                end
            end
        end
    end
    difference = rgb2gray(difference);
    difference = bwareaopen(difference,5);
    se = strel('disk',2);
    difference = imclose(difference,se);
%     bounding_box = regionprops(difference,'Boundingbox') ;
%     %imshow(img2)
%     hold on
%     for k = 1 : length(bounding_box)
%          BB = bounding_box(k).BoundingBox;
%          rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;
%     end
%difference = imabsdiff(X,Y) 
end