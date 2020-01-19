import time
import errno
import cv2
import os
from datetime import date, datetime
import numpy as np


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
    h, w = img1.shape[0], img1.shape[1]
    output = np.zeros((h, w))
    threshold = 150
    for i in range(h):
        for j in range(w):
            output[i, j] = 1 if (img2[i, j] - img1[i, j] > threshold) else 0
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.imshow("output", output)
    return connected_components(output)


def connected_components(image):
    image = image.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    # remove small components
    min_size = 1000
    img2 = np.zeros(output.shape)
    for i in range(0, nb_components - 1):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    kernel = np.ones((5, 5), np.uint8)
    # closing
    closed_image = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(closed_image.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        cv2.contourArea(c)
        x, y, w, h = rect
        cv2.rectangle(closed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.putText(closed_image, 'Moth Detected', (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
    cv2.imshow("Show", closed_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return closed_image


def main(takepics=False):
    if takepics:
        capture(2, 5)
    images = []
    directory = r'venv\Images\%s\\' % date.today().strftime("%B_%d_%Y")
    for file in os.listdir(directory):
        images.append(cv2.cvtColor(cv2.imread(directory + file), cv2.COLOR_BGR2GRAY))

    for i in range(1, len(images)):
        output = image_difference(images[i - 1], images[i])
        # cv2.imshow("final_image", output)
        # cv2.waitKey()


if __name__ == "__main__":
    main()
