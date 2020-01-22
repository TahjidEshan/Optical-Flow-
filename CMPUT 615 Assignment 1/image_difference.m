function [difference, normalized_difference, has_difference] = image_difference(frame1, frame2)
%     [rows, columns, numberOfColorChannels] = size(frame1); 
%     difference = zeros(rows, columns, numberOfColorChannels);
    normalized_difference = 0;
    threshold = 40;
    has_difference = false;
    difference = abs(double(rgb2gray(frame2))-double(rgb2gray(frame1)));
    
     difference = difference > threshold;
%     for i = 1:rows
%         for j = 1:columns
%             for k= 1:numberOfColorChannels
%                 pixeldifference = abs(double(frame2(i,j,k))-double(frame1(i,j,k)));
%                 if pixeldifference>threshold
%                     difference(i,j,k) = 1;
%                     has_difference = true;
%                 end
%             end
%         end
%     end
%     if size(difference, 3) == 3
%         difference = rgb2gray(difference);
%     end
        %imshow(difference)
%     normalized_difference = uint8(255 * difference);
%     normalized_difference = bwareaopen(normalized_difference,5);
%     se = strel('disk',2);
%     normalized_difference = imclose(normalized_difference,se);
%     bounding_box = regionprops(difference,'Boundingbox') ;
%     %imshow(img2)
%     hold on
%     for k = 1 : length(bounding_box)
%          BB = bounding_box(k).BoundingBox;
%          rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;
%     end
%difference = imabsdiff(X,Y) 
end