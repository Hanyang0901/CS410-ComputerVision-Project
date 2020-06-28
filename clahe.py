import numpy as np
from math import ceil


def interpolate(subBin, left_up, right_up, left_bottom, right_bottom, subX, subY):
    ret = np.zeros_like(subBin)
    num = subX * subY
    for i in range(subX):
        inverseI = subX - i
        for j in range(subY):
            inverseJ = subY - j
            val = subBin[i, j].astype(int)
            ret[i, j] = np.floor(
                (inverseI * (inverseJ * left_up[val] + j * right_up[val]) +
                 i * (inverseJ * left_bottom[val] + j * right_bottom[val])) / float(num))
    return ret


def process(I):
    b = I[:, :, 0]
    g = I[:, :, 1]
    r = I[:, :, 2]
    b_clahe_img = clahe(b, 2)
    g_clahe_img = clahe(g, 2)
    r_clahe_img = clahe(r, 2)
    J = np.zeros_like(I)
    J[:, :, 0] = b_clahe_img
    J[:, :, 1] = g_clahe_img
    J[:, :, 2] = r_clahe_img
    return J


def clahe(img, clipLimit, nrBins=128, nrX=0, nrY=0):
    h = img.shape[0]
    w = img.shape[1]
    if clipLimit == 1:
        return
    nrBins = max(nrBins, 128)
    if nrX == 0:
        xsz = 32
        ysz = 32
        nrX = ceil(h / xsz)
        excX = int(xsz * (nrX - h / xsz))
        nrY = ceil(w / ysz)
        excY = int(ysz * (nrY - w / ysz))
        if excX != 0:
            img = np.append(img, np.zeros((excX, img.shape[1])).astype(int), axis=0)
        if excY != 0:
            img = np.append(img, np.zeros((img.shape[0], excY)).astype(int), axis=1)
    else:
        xsz = round(h / nrX)
        ysz = round(w / nrY)
    nrPixels = xsz * ysz
    xsz2 = round(xsz / 2)
    ysz2 = round(ysz / 2)
    clahe_img = np.zeros_like(img)
    if clipLimit > 0:
        clipLimit = max(1, clipLimit * xsz * ysz / nrBins)
    else:
        clipLimit = 50

    minVal = 0
    maxVal = 255

    binSz = np.floor(1 + (maxVal - minVal) / float(nrBins))
    LUT = np.floor((np.arange(minVal, maxVal + 1) - minVal) / float(binSz))

    bins = LUT[img]
    hist = np.zeros((nrX, nrY, nrBins))
    for i in range(nrX):
        for j in range(nrY):
            bin_ = bins[i * xsz:(i + 1) * xsz, j * ysz:(j + 1) * ysz].astype(int)
            for i1 in range(xsz):
                for j1 in range(ysz):
                    hist[i, j, bin_[i1, j1]] += 1
    if clipLimit > 0:
        for i in range(nrX):
            for j in range(nrY):
                nrExcess = 0
                for nr in range(nrBins):
                    excess = hist[i, j, nr] - clipLimit
                    if excess > 0:
                        nrExcess += excess
                binIncr = nrExcess / nrBins
                upper = clipLimit - binIncr
                for nr in range(nrBins):
                    if hist[i, j, nr] > clipLimit:
                        hist[i, j, nr] = clipLimit
                    else:
                        if hist[i, j, nr] > upper:
                            nrExcess += upper - hist[i, j, nr]
                            hist[i, j, nr] = clipLimit
                        else:
                            nrExcess -= binIncr
                            hist[i, j, nr] += binIncr
                if nrExcess > 0:
                    stepSz = max(1, np.floor(1 + nrExcess / nrBins))
                    for nr in range(nrBins):
                        nrExcess -= stepSz
                        hist[i, j, nr] += stepSz
                        if nrExcess < 1:
                            break

    map_ = np.zeros((nrX, nrY, nrBins))
    scale = (maxVal - minVal) / float(nrPixels)
    for i in range(nrX):
        for j in range(nrY):
            sum_ = 0
            for nr in range(nrBins):
                sum_ += hist[i, j, nr]
                map_[i, j, nr] = np.floor(min(minVal + sum_ * scale, maxVal))
    xI = 0
    for i in range(nrX + 1):
        if i == 0:
            subX = int(xsz / 2)
            xU = 0
            xB = 0
        elif i == nrX:
            subX = int(xsz / 2)
            xU = nrX - 1
            xB = nrX - 1
        else:
            subX = xsz
            xU = i - 1
            xB = i

        yI = 0
        for j in range(nrY + 1):
            if j == 0:
                subY = int(ysz / 2)
                yL = 0
                yR = 0
            elif j == nrY:
                subY = int(ysz / 2)
                yL = nrY - 1
                yR = nrY - 1
            else:
                subY = ysz
                yL = j - 1
                yR = j
            UL = map_[xU, yL, :]
            UR = map_[xU, yR, :]
            BL = map_[xB, yL, :]
            BR = map_[xB, yR, :]
            subBin = bins[xI:xI + subX, yI:yI + subY]
            subImage = interpolate(subBin, UL, UR, BL, BR, subX, subY)
            clahe_img[xI:xI + subX, yI:yI + subY] = subImage
            yI += subY
        xI += subX
    if excX == 0 and excY != 0:
        return clahe_img[:, :-excY]
    elif excX != 0 and excY == 0:
        return clahe_img[:-excX, :]
    elif excX != 0 and excY != 0:
        return clahe_img[:-excX, :-excY]
    else:
        return clahe_img


