from tools import *


def calDepthMap(I, r):
    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:, :, 1] / 255.0
    v = hsvI[:, :, 2] / 255.0
    sigma = 0.1
    sigmaMat = np.random.normal(0, sigma, (I.shape[0], I.shape[1]))
    output = 0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat
    outputPixel = output
    output = min_filter(output, (r, r))
    outputRegion = output
    return outputRegion, outputPixel


def estA(img, J_dark):
    h, w, c = img.shape
    if img.dtype == np.uint8:
        img = np.float32(img) / 255
    n_bright = int(np.ceil(0.001 * h * w))
    reshaped_Jdark = J_dark.reshape(1, -1)
    Y = np.sort(reshaped_Jdark)
    Loc = np.argsort(reshaped_Jdark)
    Ics = img.reshape(1, h * w, 3)
    ix = img.copy()
    dx = J_dark.reshape(1, -1)
    Acand = np.zeros((1, n_bright, 3), dtype=np.float32)
    Amag = np.zeros((1, n_bright, 1), dtype=np.float32)
    for i in range(n_bright):
        x = Loc[0, h * w - 1 - i]
        ix[int(x / w), int(x % w), 0] = 0
        ix[int(x / w), int(x % w), 1] = 0
        ix[int(x / w), int(x % w), 2] = 1
        Acand[0, i, :] = Ics[0, Loc[0, h * w - 1 - i], :]
        Amag[0, i] = np.linalg.norm(Acand[0, i, :])
    reshaped_Amag = Amag.reshape(1, -1)
    Y2 = np.sort(reshaped_Amag)
    Loc2 = np.argsort(reshaped_Amag)
    if len(Y2) > 20:
        A = Acand[0, Loc2[0, n_bright - 19:n_bright], :]
    else:
        A = Acand[0, Loc2[0, n_bright - len(Y2):n_bright], :]
    return A


def process(I):
    r = 7
    beta = 1.0
    filter_r = 60
    eps = 0.01
    region, pixel = calDepthMap(I, r)
    guided_filter = GuidedFilter(I, filter_r, eps)
    refineDR = guided_filter.filter(region)
    tR = np.exp(-beta * refineDR)
    tP = np.exp(-beta * pixel)
    a = estA(I, region)
    I = np.float32(I) / 255
    h, w, c = I.shape
    J = np.zeros((h, w, c), dtype=np.float32)
    J[:, :, 0] = I[:, :, 0] - a[0, 0]
    J[:, :, 1] = I[:, :, 1] - a[0, 1]
    J[:, :, 2] = I[:, :, 2] - a[0, 2]
    t = tR
    t0, t1 = 0.05, 1
    t = t.clip(t0, t1)
    J[:, :, 0] = J[:, :, 0] / t
    J[:, :, 1] = J[:, :, 1] / t
    J[:, :, 2] = J[:, :, 2] / t
    J[:, :, 0] = J[:, :, 0] + a[0, 0]
    J[:, :, 1] = J[:, :, 1] + a[0, 1]
    J[:, :, 2] = J[:, :, 2] + a[0, 2]
    return J
