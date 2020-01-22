function [] = opticalflow(frame1, frame2, blocksize)
    image1gray = im2double(rgb2gray(frame1));
    image1 = imresize(image1gray, 0.5); % downsize to half
    image2gray = im2double(rgb2gray(frame2));
    image2 = imresize(image2gray, 0.5); % downsize to half
    w = round(blocksize/2);
    [Ix_m, Iy_m] = gradient(image1);
    It_m =  image_difference(frame1, frame2);
    u = zeros(size(image1));
    v = zeros(size(image2));

    % within window ww * ww
    for i = w+1:size(Ix_m,1)-w
       for j = w+1:size(Ix_m,2)-w
          Ix = Ix_m(i-w:i+w, j-w:j+w);
          Iy = Iy_m(i-w:i+w, j-w:j+w);
          It = It_m(i-w:i+w, j-w:j+w);
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
       end;
    end;

    u_deci = u(1:10:end, 1:10:end);
    v_deci = v(1:10:end, 1:10:end);
    [m, n] = size(image1gray);
    [X,Y] = meshgrid(1:n, 1:m);
    X_deci = X(1:20:end, 1:20:end);
    Y_deci = Y(1:20:end, 1:20:end);
    imshow(frame2);
    hold on;
    % draw the velocity vectors
    quiver(X_deci, Y_deci, u_deci,v_deci, 'y')
end

