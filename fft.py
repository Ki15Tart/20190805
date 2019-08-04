# -*- coding: utf-8 -*-
#マウス操作　完成版
import cv2
import numpy as np
from numpy.random import rand
from numpy import uint8, float32, float64, log, pi, sin, cos, abs, sqrt
import matplotlib.pyplot as plt
from PIL import Image

# マウスイベント時に処理を行う
def mouse_event(event, x, y, flags, param):
    global draw

    #左クリックがあったら表示
    if event == cv2.EVENT_LBUTTONDOWN: # 左ボタンを押下したとき
        draw = True
    if event == cv2.EVENT_LBUTTONUP: # 左ボタンを上げたとき
        draw = False
    if event == cv2.EVENT_MOUSEMOVE: # マウスが動いた時
        if draw:
            img2[:, :] = 0
            img2[y:y+4, x:x+4] = 1
            img[y:y+4, x:x+4] = 1

            
image = cv2.imread("fuji_g.jpg", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("original")

fimage = np.fft.fft2(image) 
# Replace quadrant
# 1st <-> 3rd, 2nd <-> 4th
fimg =  np.fft.fftshift(fimage)
# Power spectrum calculation
mag = 20*np.log(np.abs(fimg))
mag_max = int(np.max(mag))
mag_min = int(np.min(mag))
mag2 = ((mag - mag_min)/(mag_max - mag_min)*255).astype(uint8)
cv2.namedWindow("spectrum")

# 画像の読み込み
img = np.zeros(image.shape)
# ウィンドウのサイズを変更可能にする
cv2.namedWindow("zahyo")
# マウスイベント時に関数mouse_eventの処理を行う
cv2.setMouseCallback("zahyo", mouse_event)

# 画像の読み込み
img2 = np.zeros(image.shape)
# ウィンドウのサイズを変更可能にする
cv2.namedWindow("sin")

cv2.namedWindow("ifft")

draw = False

# 「Q」が押されるまで画像を表示する
while (True):
    fsin_shift = np.fft.fftshift(fimg*img2)
    fsin_ss = np.fft.ifft2(fsin_shift)
    fsin_ss = np.abs(fsin_ss)
    fsin_max = np.max(fsin_ss)
    fsin_min = np.min(fsin_ss)
    fsin = ((fsin_ss - fsin_min)/(fsin_max - fsin_min)*255).astype(uint8)
    
    fsin1_shift = np.fft.fftshift(fimg*img)
    fsin1_ss = np.fft.ifft2(fsin1_shift)
    fsin1_ss = np.abs(fsin1_ss)
    fsin1_max = np.max(fsin1_ss)
    fsin1_min = np.min(fsin1_ss)
    fsin1 = ((fsin1_ss - fsin1_min)/(fsin1_max - fsin1_min)*255).astype(uint8)
    
    cv2.imshow("original", image)
    cv2.imshow("spectrum", mag2)
    cv2.imshow("zahyo", img)
    cv2.imshow("sin", fsin)
    cv2.imshow("ifft", fsin1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()