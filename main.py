import cv2
import darkchannel
import clahe
import colorAttenuation

if __name__ == '__main__':
    filename = "input.png"
    ori_img = cv2.imread(filename)
    img1 = darkchannel.process(ori_img)
    img2 = clahe.process(ori_img)
    img3 = colorAttenuation.process(ori_img)
    cv2.imshow("dark channel prior", img1)
    cv2.imshow("CLAHE", img2)
    cv2.imshow("color attenuation", img3)
    cv2.waitKey()
