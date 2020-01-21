import time
import errno
import cv2
import os
from datetime import date, datetime
import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt


# from PIL import Image


def capture(num_frames, delay):
    camera = cv2.VideoCapture(0)
    directory = r'venv\Images\%s' % date.today().strftime("%B_%d_%Y")
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            print("Creation of the directory %s failed" % directory)
            if exc.errno != errno.EEXIST:
                raise
        else:
            print("Successfully created the directory %s" % directory)
    for i in range(num_frames):
        return_value, image = camera.read()
        imagename = r'{}\capture_{}_at_{}.png'.format(directory, i, datetime.now().strftime("%H_%M_%S"))
        cv2.imwrite(imagename, image)
        time.sleep(delay)
    del camera


def image_difference(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape[0], img1.shape[1]
    output = np.zeros((h, w))
    threshold = 100
    for i in range(h):
        for j in range(w):
            output[i, j] = 1 if (img2[i, j] - img1[i, j] > threshold) else 0
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.imshow("output", output)
    return connected_components(output)


def connected_components(image, show=False, draw_bounding_box=False):
    image = image.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    # remove small components
    min_size = 1000
    img2 = np.zeros(output.shape)
    for i in range(0, nb_components - 1):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
    kernel = np.ones((5, 5), np.uint8)
    # closing
    closed_image = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(closed_image.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if draw_bounding_box:
        for c in contours:
            rect = cv2.boundingRect(c)
            cv2.contourArea(c)
            x, y, w, h = rect
            cv2.rectangle(closed_image, (x, y), (x + w, y + h), (1, 0, 0), 2)
            # cv2.putText(closed_image, 'Moth Detected', (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
    if show:
        cv2.imshow("Show", closed_image)
        cv2.waitKey(500)
        # cv2.destroyAllWindows()

    return closed_image, contours


def Lucas_Kanade(oldframe, newframe):
    I1 = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)

    I2 = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)

    color = np.random.randint(0, 255, (100, 3))
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))

    Ix = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2
    Iy = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2
    It1 = convolve2d(I1, Gt1) + convolve2d(I2,
                                           Gt2)

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    features = cv2.goodFeaturesToTrack(I1, mask=None,
                                       **feature_params)
    print("here",features)
    feature = np.int32(features)
    feature = np.reshape(feature, newshape=[-1, 2])

    u = np.ones(Ix.shape)
    v = np.ones(Ix.shape)
    status = np.zeros(feature.shape[0])
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    mask = np.zeros_like(oldframe)

    newFeature = np.zeros_like(feature)

    for a, i in enumerate(feature):

        x, y = i

        A[0, 0] = np.sum((Ix[y - 1:y + 2, x - 1:x + 2]) ** 2)

        A[1, 1] = np.sum((Iy[y - 1:y + 2, x - 1:x + 2]) ** 2)
        A[0, 1] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
        A[1, 0] = np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * Iy[y - 1:y + 2, x - 1:x + 2])
        Ainv = np.linalg.pinv(A)

        B[0, 0] = -np.sum(Ix[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
        B[1, 0] = -np.sum(Iy[y - 1:y + 2, x - 1:x + 2] * It1[y - 1:y + 2, x - 1:x + 2])
        prod = np.matmul(Ainv, B)

        u[y, x] = prod[0]
        v[y, x] = prod[1]

        newFeature[a] = [np.int32(x + u[y, x]), np.int32(y + v[y, x])]
        if np.int32(x + u[y, x]) == x and np.int32(
                y + v[y, x]) == y:
            status[a] = 0
        else:
            status[a] = 1

    um = np.flipud(u)
    vm = np.flipud(v)

    good_new = newFeature[
        status == 1]
    good_old = feature[status == 1]
    print(good_new.shape)
    print(good_old.shape)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        newframe = cv2.circle(newframe, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(newframe, mask)
    return img


def lucas_kanade_np(im1, im2, win=2):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    assert im1.shape == im2.shape
    I_x = np.zeros(im1.shape)
    I_y = np.zeros(im1.shape)
    I_t = np.zeros(im1.shape)
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    params = np.zeros(im1.shape + (5,))  # Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x  # I_x2
    params[..., 1] = I_y * I_y  # I_y2
    params[..., 2] = I_x * I_y  # I_xy
    params[..., 3] = I_x * I_t  # I_xt
    params[..., 4] = I_y * I_t  # I_yt
    del I_x, I_y, I_t
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(im1.shape + (2,))
    det = win_params[..., 0] * win_params[..., 1] - win_params[..., 2] ** 2
    op_flow_x = np.where(det != 0,
                         (win_params[..., 1] * win_params[..., 3] -
                          win_params[..., 2] * win_params[..., 4]) / det,
                         0)
    op_flow_y = np.where(det != 0,
                         (win_params[..., 0] * win_params[..., 4] -
                          win_params[..., 2] * win_params[..., 3]) / det,
                         0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
    return op_flow


def optical_flow(images, blocksize=16):
    threshold = .001
    for i in range(1, len(images)):
        last_frame = cv2.cvtColor(images[i - 1], cv2.COLOR_BGR2GRAY)
        current_frame = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        h, w = current_frame.shape[0], current_frame.shape[1]
        mask = np.zeros((h, w))
        for r in range(0, current_frame.shape[0] - blocksize, blocksize):
            for c in range(0, current_frame.shape[1] - blocksize, blocksize):
                last_window = last_frame[r:r + blocksize, c:c + blocksize]
                current_window = current_frame[r:r + blocksize, c:c + blocksize]
                block_difference = current_window - last_window
                if not (block_difference < threshold).all():
                    mask[r: r + blocksize, c: c + blocksize] = block_difference
        mask, contours = connected_components(mask)
        for c in contours:
            rect = cv2.boundingRect(c)
            cv2.contourArea(c)
            x, y, w, h = rect
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (1, 0, 0), 2)
        cv2.imshow("here", mask)
        cv2.waitKey(500)


def compute_flow_map(u, v, gran=8):
    flow_map = np.zeros(u.shape)

    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):

            if y % gran == 0 and x % gran == 0:
                dx = 10 * int(u[y, x])
                dy = 10 * int(v[y, x])

                if dx > 0 or dy > 0:
                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 255, 1)

    return flow_map


def lucas_kanade(im1, im2, win=7):
    Ix = np.zeros(im1.shape)
    Iy = np.zeros(im1.shape)
    It = np.zeros(im1.shape)

    Ix[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]

    params = np.zeros(im1.shape + (5,))
    params[..., 0] = cv2.GaussianBlur(Ix * Ix, (5, 5), 3)
    params[..., 1] = cv2.GaussianBlur(Iy * Iy, (5, 5), 3)
    params[..., 2] = cv2.GaussianBlur(Ix * Iy, (5, 5), 3)
    params[..., 3] = cv2.GaussianBlur(Ix * It, (5, 5), 3)
    params[..., 4] = cv2.GaussianBlur(Iy * It, (5, 5), 3)

    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    Ixx = win_params[..., 0]
    Iyy = win_params[..., 1]
    Ixy = win_params[..., 2]
    Ixt = -win_params[..., 3]
    Iyt = -win_params[..., 4]

    M_det = Ixx * Iyy - Ixy ** 2
    temp_u = Iyy * (-Ixt) + (-Ixy) * (-Iyt)
    temp_v = (-Ixy) * (-Ixt) + Ixx * (-Iyt)
    op_flow_x = np.where(M_det != 0, temp_u / M_det, 0)
    op_flow_y = np.where(M_det != 0, temp_v / M_det, 0)

    u[win + 1: -1 - win, win + 1: -1 - win] = op_flow_x[:-1, :-1]
    v[win + 1: -1 - win, win + 1: -1 - win] = op_flow_y[:-1, :-1]

    return u, v


def lkbuiltin(images):
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    old_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(images[0])
    color = np.random.randint(0, 255, (100, 3))
    for i in range(1, len(images)):
        # output, contours = image_difference(images[0], images[i])
        frame_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[j].tolist(), 2)
            frame = cv2.circle(images[i], (a, b), 5, color[j].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        cv2.waitKey(500)


def main(takepics=False):
    if takepics:
        capture(6, 1)
    images = []
    # directory = r'venv\Images\%s\\' % date.today().strftime("%B_%d_%Y")
    directory = r'venv\Images\armD32im1\\'

    for file in os.listdir(directory):
        images.append(cv2.imread(directory + file))

    for i in range(1, len(images)):
        img = Lucas_Kanade(images[0], images[i])
        cv2.imshow("img", img)
        cv2.waitKey(500)
    # optical_flow(images)
    # firstframe = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    # for i in range(1, len(images)):
    #     currentframe = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    #     u, v = lucas_kanade(firstframe, currentframe)
    #     flow = compute_flow_map(u, v)
    #     # plt.imshow(flow, cmap='gray')
    #     # plt.show()
    #     mask = currentframe+flow
    #     plt.imshow(mask, cmap='gray')
    #     plt.show()
    #     # cv2.imshow("final", mask)
    #     # cv2.waitKey(500)


if __name__ == "__main__":
    main()
