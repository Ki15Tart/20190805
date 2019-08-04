# 目的

画像にフーリエ変換を行い, マウスで周波数を指定し正弦波を取得して重ね合わせることで, 逆フーリエ変換の動作をデモンストレーションするプログラム

## 動作環境

Anacondaをインストール

python 3.7

opencv 4.0.1

Jupyter　Notebookでプログラムを実行

## ソースコード

```python
#モジュールのインポート
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
            

#元画像
image = cv2.imread("fuji_g.jpg", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("original")

#スペクトラム
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

#マウス操作から座標を取得するための画像
img = np.zeros(image.shape)
cv2.namedWindow("zahyo")
cv2.setMouseCallback("zahyo", mouse_event)  #マウスによるイベント処理

#取得した座標の正弦波を表す画像
img2 = np.zeros(image.shape)
cv2.namedWindow("sin")

#正弦波を重ねて表示する画像
cv2.namedWindow("ifft")

draw = False


# 「Q」が押されるまで処理を行う
while (True):
    #各座標の正弦波を計算
    fsin_shift = np.fft.fftshift(fimg*img2)
    fsin_ss = np.fft.ifft2(fsin_shift)
    fsin_ss = np.abs(fsin_ss)
    fsin_max = np.max(fsin_ss)
    fsin_min = np.min(fsin_ss)
    fsin = ((fsin_ss - fsin_min)/(fsin_max - fsin_min)*255).astype(uint8)
    
    #正弦波を重ねたものを計算
    fsin1_shift = np.fft.fftshift(fimg*img)
    fsin1_ss = np.fft.ifft2(fsin1_shift)
    fsin1_ss = np.abs(fsin1_ss)
    fsin1_max = np.max(fsin1_ss)
    fsin1_min = np.min(fsin1_ss)
    fsin1 = ((fsin1_ss - fsin1_min)/(fsin1_max - fsin1_min)*255).astype(uint8)
    
    #生成したすべての画像をウィンドウに出力
    cv2.imshow("original", image)
    cv2.imshow("spectrum", mag2)
    cv2.imshow("zahyo", img)
    cv2.imshow("sin", fsin)
    cv2.imshow("ifft", fsin1)
    
    #プログラムの終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
```

## 動作の概要
1. 元画像を読み込む
2. フーリエ変換を行い, パワースペクトラムを計算
3. 元画像と同サイズの真っ黒な画像を生成し, マウスで周波数を指定して白く変化させる
4. 各周波数に対し, フーリエ逆変換を行い正弦波を出力
5. 正弦波を重ね, 元の画像に近づけていく


## 参考資料

フーリエ変換・スペクトラムの計算： https://www.hello-python.com/2018/02/16/numpyとopencvを使った画像のフーリエ変換と逆変換

マウスイベント： http://rasp.hateblo.jp/entry/2016/01/24/204539

マウスのドラッグ処理： https://ensekitt.hatenablog.com/entry/2018/06/17/200000

### 動作のスクリーンショットをgif動画にしているものをアップロードしている（fft.gif）
