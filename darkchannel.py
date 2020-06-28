import math
from tools import *


def DarkChannel(im, radius):
    b, g, r = cv2.split(im)
    minvalue = cv2.min(cv2.min(r, g), b)
    dark_channel = np.zeros(minvalue.shape)
    for i in range(im.shape[0]):
        left = i - radius if radius < i else 0
        right = i + radius + 1
        if right > im.shape[0]:
            right = im.shape[0]
        for j in range(im.shape[1]):
            up = j - radius if radius < j else 0
            down = j + radius + 1
            if down > im.shape[1]:
                down = im.shape[1]
            dark_channel[i][j] = np.min(minvalue[left:right, up:down])
    return dark_channel


def getA(im, dark):
    h, w = im.shape[:2]
    imSize = h * w
    num = int(max(math.floor(imSize / 1000), 1))
    dark_channel = dark.reshape(imSize, 1)
    rgb_channel = im.reshape(imSize, 3)
    indices = dark_channel.argsort()
    indices = indices[imSize - num:]
    A = np.average(rgb_channel[indices, :], axis=0)
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    guided_filter = GuidedFilter(im, r, eps)
    t = guided_filter.filter(et)
    return t


def recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
    return res


def process(img):
    I = img.astype('float64') / 255
    dark = DarkChannel(I, 7)
    A = getA(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(img, te)
    J = recover(I, t, A, 0.1)
    return J
