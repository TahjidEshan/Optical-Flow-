function [] = opticalflownew(frame1, frame2, blocksize)
    frame1gray = im2double(rgb2gray(frame1));
    frame1resized = imresize(frame1gray, 0.5); 
    im2tframe2gray = im2double(rgb2gray(frame2));
    frame2resized = imresize(im2tframe2gray, 0.5);
    window = round(blocksize/2);

    Ix_m = conv2(frame1resized,[-1 1; -1 1], 'valid'); 
    Iy_m = conv2(frame1resized, [-1 -1; 1 1], 'valid'); 
    It_m = conv2(frame1resized, ones(2), 'valid') + conv2(frame2resized, -ones(2), 'valid');
    u = zeros(size(frame1resized));
    v = zeros(size(frame2resized));

    for i = window+1:size(Ix_m,1)-window
       for j = window+1:size(Ix_m,2)-window
          Ix = Ix_m(i-window:i+window, j-window:j+window);
          Iy = Iy_m(i-window:i+window, j-window:j+window);
          It = It_m(i-window:i+window, j-window:j+window);

          Ix = Ix(:);
          Iy = Iy(:);
          b = -It(:); 

          A = [Ix Iy]; 
          V = pinv(A)*b; 

          u(i,j)=V(1);
          v(i,j)=V(2);
       end
    end


    u_final = u(1:10:end, 1:10:end);
    v_final = v(1:10:end, 1:10:end);
    [rows, columns] = size(frame1gray);
    [X,Y] = meshgrid(1:columns, 1:rows);
    X_final = X(1:20:end, 1:20:end);
    Y_final = Y(1:20:end, 1:20:end);
    subplot(2,2,1);
    imshow(frame1);
    subplot(2,2,2);
    imshow(frame2);
    subplot(2,2,3);    
    imshow(uint8(image_difference(frame1, frame2)), 'DisplayRange', []);
    subplot(2,2,4);
    imshow(frame2);
    hold on;
    quiver(X_final, Y_final, u_final,v_final, 'r')
end