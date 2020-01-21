import time
import errno
import cv2
import os
from datetime import date, datetime
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy.linalg as lin


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


def gaussian_kernel():
    h1 = 15
    h2 = 15
    x, y = np.mgrid[0:h2, 0:h1]
    x = x - h2 / 2
    y = y - h1 / 2
    sigma = 1.5
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def derivative(frame1, frame2):
    g = gaussian_kernel()
    img_smooth = signal.convolve(frame1, g, mode='same')
    fx, fy = np.gradient(img_smooth)
    ft = signal.convolve2d(frame1, 0.25 * np.ones((2, 2))) + \
         signal.convolve2d(frame2, -0.25 * np.ones((2, 2)))

    fx = fx[0:fx.shape[0] - 1, 0:fx.shape[1] - 1]
    fy = fy[0:fy.shape[0] - 1, 0:fy.shape[1] - 1];
    ft = ft[0:ft.shape[0] - 1, 0:ft.shape[1] - 1];
    return fx, fy, ft


def opticalflow_lucas_kanade(frame1, frame2, i, j, window_size):
    fx, fy, ft = derivative(frame1, frame2)
    window = np.floor(window_size / 2)
    Fx = fx[i - window - 1:i + window,
         j - window - 1:j + window]
    Fy = fy[i - window - 1:i + window,
         j - window - 1:j + window]
    Ft = ft[i - window - 1:i + window,
         j - window - 1:j + window]
    Fx = Fx.T
    Fy = Fy.T
    Ft = Ft.T

    Fx = Fx.flatten(order='F')
    Fy = Fy.flatten(order='F')
    Ft = -Ft.flatten(order='F')

    A = np.vstack((Fx, Fy)).T
    U = np.dot(np.dot(lin.pinv(np.dot(A.T, A)), A.T), Ft)
    return U[0], U[1]


# def optical_flow(prev_frame, current_frame, windowsize=4):
# feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
#
# color = np.random.randint(0, 255, (100, 3))
# old_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#
# points0 = cv2.goodFeaturesToTrack(old_frame_gray, mask=None, **feature_params)
# feature_point = [p.ravel() for p in points0]
# feature_point = feature_point[:1]
#
# mask = np.zeros_like(old_frame_gray)
# current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#
# gX = cv2.Sobel(old_frame_gray, cv2.CV_32F, 1, 0, ksize=5)
# gY = cv2.Sobel(old_frame_gray, cv2.CV_32F, 0, 1, ksize=5)
# good_new = [get_New_Coordinate(old_frame_gray, current_frame_gray, int(x), int(y), 15, gX, gY) for x, y in
#             feature_point]
# newfeature_point = []

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

def main(takepics=False):
    if takepics:
        capture(6, 1)
    images = []
    # directory = r'venv\Images\%s\\' % date.today().strftime("%B_%d_%Y")
    directory = r'venv\Images\armD32im1\\'

    for file in os.listdir(directory):
        images.append(cv2.imread(directory + file))

    for i in range(1, len(images)):
        old_frame = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        current_frame = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        u, v = opticalflow_lucas_kanade(old_frame,current_frame, window_size=16)
        flow = compute_flow_map(u, v)
        mask = current_frame+flow
        plt.imshow(mask, cmap='gray')
        plt.show()

if __name__ == "__main__":
    main()
