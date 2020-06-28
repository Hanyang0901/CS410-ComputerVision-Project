import cv2
import numpy as np


def min_filter(img, size):
    r1 = size[0]
    r2 = size[1]
    ret = np.zeros_like(img)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            left = j - r1
            if left < 0:
                left = 0
            right = j + r1 + 1
            if right > img.shape[1]:
                right = ret.shape[1]
            up = i - r2
            if up < 0:
                up = 0
            down = i + r2 + 1
            if down > ret.shape[0]:
                down = ret.shape[0]
            ret[i, j] = np.min(img[up:down, left:right])
    return ret


# the implemented mid_filter is ok to use, but it will take much more time compared to cv2.blur.
# If you are sure to use this, replace all cv2.blur with mid_filter
def mid_filter(img, size):
    r1 = size[0]
    r2 = size[1]
    ret = np.zeros_like(img)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            left = j - r1
            if left < 0:
                left = 0
            right = j + r1 + 1
            if right > img.shape[1]:
                right = ret.shape[1]
            up = i - r2
            if up < 0:
                up = 0
            down = i + r2 + 1
            if down > ret.shape[0]:
                down = ret.shape[0]
            ret[i, j] = np.average(img[up:down, left:right])
    return ret


def computeOutput(ab, I):
    ar_mean, ag_mean, ab_mean, b_mean = ab
    Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
    q = ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean
    return q


def toFloatImg(img):
    if img.dtype == np.float32:
        return img
    return img.astype(np.float32) / 255


class GuidedFilter:
    def __init__(self, I, radius=5, epsilon=0.4):
        radius = 2 * radius + 1
        self.radius = radius
        self.epsilon = epsilon
        self.I = toFloatImg(I)
        I = self.I

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (radius, radius))
        self._Ig_mean = cv2.blur(Ig, (radius, radius))
        self._Ib_mean = cv2.blur(Ib, (radius, radius))

        Irr_var = cv2.blur(Ir ** 2, (radius, radius)) - self._Ir_mean ** 2 + epsilon
        Irg_var = cv2.blur(Ir * Ig, (radius, radius)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (radius, radius)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig * Ig, (radius, radius)) - self._Ig_mean * self._Ig_mean + epsilon
        Igb_var = cv2.blur(Ig * Ib, (radius, radius)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib * Ib, (radius, radius)) - self._Ib_mean * self._Ib_mean + epsilon

        self._Ir_mean = cv2.blur(Ir, (radius, radius))
        self._Ig_mean = cv2.blur(Ig, (radius, radius))
        self._Ib_mean = cv2.blur(Ib, (radius, radius))

        Irr_var = cv2.blur(Ir ** 2, (radius, radius)) - self._Ir_mean ** 2 + epsilon
        Irg_var = cv2.blur(Ir * Ig, (radius, radius)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (radius, radius)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig * Ig, (radius, radius)) - self._Ig_mean * self._Ig_mean + epsilon
        Igb_var = cv2.blur(Ig * Ib, (radius, radius)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib * Ib, (radius, radius)) - self._Ib_mean * self._Ib_mean + epsilon

        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        Irr_inv /= I_cov
        Irg_inv /= I_cov
        Irb_inv /= I_cov
        Igg_inv /= I_cov
        Igb_inv /= I_cov
        Ibb_inv /= I_cov

        self._Irr_inv = Irr_inv
        self._Irg_inv = Irg_inv
        self._Irb_inv = Irb_inv
        self._Igg_inv = Igg_inv
        self._Igb_inv = Igb_inv
        self._Ibb_inv = Ibb_inv

    def computeCoefficients(self, p):
        r = self.radius
        I = self.I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        p_mean = cv2.blur(p, (r, r))
        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean

        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov

        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        ar_mean = cv2.blur(ar, (r, r))
        ag_mean = cv2.blur(ag, (r, r))
        ab_mean = cv2.blur(ab, (r, r))
        b_mean = cv2.blur(b, (r, r))

        return ar_mean, ag_mean, ab_mean, b_mean

    def filter(self, p):
        p_32F = toFloatImg(p)
        ab = self.computeCoefficients(p)
        return computeOutput(ab, self.I)
