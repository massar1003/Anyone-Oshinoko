import cv2
import numpy as np

# create the haar cascade
eye_cascade = cv2.CascadeClassifier('sample/07/haarcascade_eye.xml')

# ファイルの読み込み
read = 'pic/Ai_eye.png'
file1 = cv2.imread(read, -1)
file2 = cv2.imread("pic/AbeHiroshi.png", -1)
file3 = cv2.cvtColor(file2, cv2.COLOR_BGR2GRAY)
file4 = cv2.imread("pic/AbeHiroshi.png", -1)

file1_width = file1.shape[0]
file1_height = file1.shape[1]
out = np.zeros(file1.shape[:4], np.uint8)

file4_height = file4.shape[0]
file4_width = file4.shape[1]

# 透過の処理
for i in range(file1_width):
    for j in range(file1_height):
        b = file1[i][j][0]
        r = file1[i][j][1]
        g = file1[i][j][2]
        a = file1[i][j][3]
        if b>0 or r>0 or g>0:
            out[i][j][0] = 255
            out[i][j][1] = 255
            out[i][j][2] = 255
            out[i][j][3] = 255
        else:
            out[i][j][0] = b
            out[i][j][1] = r
            out[i][j][2] = g
            out[i][j][3] = 0

# 目を検出
detected_eyes = eye_cascade.detectMultiScale(file3, 1.1, 3)

# 検出した目の数を表示
print("Found {0} eyes!".format(len(detected_eyes)))

lower_range = np.array([0, 0, 0], dtype=np.uint8)  # 背景の下限値
upper_range = np.array([255, 255, 255], dtype=np.uint8)  # 背景の上限値

# 目に画像を張り付ける
for (x, y, w, h) in detected_eyes:
    resize = cv2.resize(out, (w, h))
    file2[y:y+h, x:x+w] = resize
    print(w)
    print(h)

#最後に背景として重ねる画像
img = np.zeros((file4_height, file4_width, 4), np.uint8)

#背景画像と重ね合わせることで透過したときの後ろの画像が映るようにしている。
img = cv2.addWeighted(file2, 1.0, file4, 1.0, 0.0)

cv2.imshow("a",img)
cv2.imwrite("test1.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#out,resizeは出力画像で確認すると画像透過できている。
