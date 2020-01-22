function [] = opticalflow(frame1, frame2, blockSize)
    window = round(blockSize/2);
    im1 = double(rgb2gray(frame1));
    im2 = double(rgb2gray(frame2));
    [rows, columns] = size(im1);
    [diX, diY] = gradient(im1);
    [difference, ~, ~] = image_difference(frame1, frame2);
    u = zeros(size(im1));
    v = zeros(size(im2));
    for i=  window+1:size(diX,1)-window
       for j = window+1:size(diX,2)-window
            Ix = diX(i-window:i+window, j-window:j+window);
            Iy = diY(i-window:i+window, j-window:j+window);
            It = difference(i-window:i+window, j-window:j+window);
            a11 = Ix.*Ix;
            a12 = Ix.*Iy;
            a22 = Iy.*Iy;
            b11 = Ix.*It;
            b12 = Iy.*It;
            A11 = sum(a11(:));
            A12 = sum(a12(:));
            A21 = sum(a12(:));
            A22 = sum(a22(:));
            B11 = sum(b11(:));
            B12 = sum(b12(:));
            A = [A11 A12; A21 A22];
            B = [B11; B12];
%             Ix = Ix(:);
%             Iy = Iy(:);
%             b = -It(:); 
% 
%             A = [Ix Iy]; 
            nu = -pinv(A)*B; 

            u(i,j)=nu(1);
            v(i,j)=nu(2);
       end
     end
%     X = (1 : blockSize : columns-1) + blockSize/2;
%     Y = (1 : blockSize : rows-1) + blockSize/2;
    [X,Y] = meshgrid(1: columns, 1: rows);
%     disp(u)
%     figure;
%     close all
    imshow(frame2)
    hold on
    quiver(X(1:20:end, 1:20:end), Y(1:20:end,1:20:end), u(1:20:end, 1:20:end), v(1:20:end, 1:20:end))
%     pause(.5)
    hold off
%     close all
end