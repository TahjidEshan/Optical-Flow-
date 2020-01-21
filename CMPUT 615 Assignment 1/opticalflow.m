function [] = opticalflow(frame1, frame2, blocksize)
    sc = 2;
    im1t = im2double(rgb2gray(frame1));
    im1 = imresize(im1t, 1/sc); 
    im2t = im2double(rgb2gray(frame2));
    im2 = imresize(im2t, 1/sc); 
    
    w = round(blocksize/2);


    Ix_m = conv2(im1,[-1 1; -1 1], 'valid'); 
    Iy_m = conv2(im1, [-1 -1; 1 1], 'valid');
    It_m = conv2(im1, ones(2), 'valid') + conv2(im2, -ones(2), 'valid'); 
    u = zeros(size(im1));
    v = zeros(size(im2));


    for i = w+1:size(Ix_m,1)-w
       for j = w+1:size(Ix_m,2)-w
          Ix = Ix_m(i-w:i+w, j-w:j+w);
          Iy = Iy_m(i-w:i+w, j-w:j+w);
          It = It_m(i-w:i+w, j-w:j+w);

          Ix = Ix(:);
          Iy = Iy(:);
          b = -It(:); 

          A = [Ix Iy]; 
          nu = pinv(A)*b; 

          u(i,j)=nu(1);
          v(i,j)=nu(2);
       end
    end

    u_deci = u(1:10:end, 1:10:end);
    v_deci = v(1:10:end, 1:10:end);
    [m, n] = size(im1t);
    [X,Y] = meshgrid(1:n, 1:m);
    X_deci = X(1:20:end, 1:20:end);
    Y_deci = Y(1:20:end, 1:20:end);
    figure();
    imshow(frame2);
    hold on;
    quiver(X_deci, Y_deci, u_deci,v_deci, 'y')
    pause(.5)
    close all