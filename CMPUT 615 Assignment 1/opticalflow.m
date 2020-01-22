function [] = opticalflow(frame1, frame2, blocksize)
    image1gray = im2double(rgb2gray(frame1));
    image1 = imresize(image1gray, 0.5); 
    image2gray = im2double(rgb2gray(frame2));
    image2 = imresize(image2gray, 0.5); 
    window = round(blocksize/2);
    [Ix_m, Iy_m] = gradient(image1);
    It_m =  image_difference(frame1, frame2);
    u = zeros(size(image1));
    v = zeros(size(image2));

    for i = window+1:size(Ix_m,1)-window
       for j = window+1:size(Ix_m,2)-window
          Ix = Ix_m(i-window:i+window, j-window:j+window);
          Iy = Iy_m(i-window:i+window, j-window:j+window);
          It = It_m(i-window:i+window, j-window:j+window);
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
          b = -[B11; B12];
          nu = pinv(A)*b; 
          u(i,j)=nu(1);
          v(i,j)=nu(2);
       end
    end

    u_final = u(1:10:end, 1:10:end);
    v_final = v(1:10:end, 1:10:end);
    [rows, columns] = size(image1gray);
    [X,Y] = meshgrid(1:columns, 1:rows);
    X_final = X(1:20:end, 1:20:end);
    Y_final = Y(1:20:end, 1:20:end);
            subplot(2,2,1);
            imshow(frame1);
            subplot(2,2,2);
            imshow(frame2);
            subplot(2,2,3);    
            imshow(uint8(It_m), 'DisplayRange', []);
            subplot(2,2,4);
    imshow(frame2);
    hold on;
    % draw the velocity vectors
    quiver(X_final, Y_final, u_final,v_final, 'y')
end

