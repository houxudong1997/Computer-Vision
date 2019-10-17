import cv2
import glob
import numpy as np

cbrow = 35
cbcol = 35

objp = np.zeros((cbrow*cbcol,3), np.float32)

# 设定世界坐标下点的坐标值，因为用的是棋盘可以直接按网格取；
# 假定棋盘正好在x-y平面上，这样z=0，简化初始化步骤。
# mgrid把列向量[0:cbraw]复制了cbcol列，把行向量[0:cbcol]复制了cbraw行。
# 转置reshape后，每行都是35*35网格中的某个点的坐标。

objp[:,:2] = np.mgrid[0:cbrow,0:cbcol].T.reshape(-1,2)
print(objp)
objpoints = []
imgpoints = []

images = glob.glob("D:\PyCharmProjects\calibrate\*.jpg")
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('img',gray)
    #cv2.waitKey(1000)
    ret, corners = cv2.findChessboardCorners(gray,(35,35),None)
    #criteria:角点精准化迭代过程的终止条件
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #执行亚像素级角点检测
    #corners2 = cv2.cornerSubPix(gray,corners,(36,36),(-1,-1),criteria)#(11,11)
    objpoints.append(objp)

    imgpoints.append(corners)
    #在棋盘上绘制角点,只是可视化工具
    img = cv2.drawChessboardCorners(gray,(35,35),corners,ret)
    # cv2.imshow('img',img)
    # cv2.waitKey(1000)

# mtx，相机内参；dist，畸变系数；revcs，旋转矩阵；tvecs，平移矩阵。

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print ("mtx:\n",mtx)

img = cv2.imread('D:\PyCharmProjects\calibrate\ca1.jpg')
img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
h,w = img.shape[:2]
# 优化相机内参（camera matrix）
# 参数1表示保留所有像素点，同时可能引入黑色像素
# 设为0表示尽可能裁剪不想要的像素，这是个scale，0-1都可以取

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# 纠正畸变
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('ceresult.png',dst)

# cv2.imshow('dst',dst)
# cv2.waitKey(1000)
# 打印我们要求的两个矩阵参数
print ("newcameramtx:\n",newcameramtx)
print ("dist:\n",dist)
# 计算误差
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print ("total error: ", tot_error/len(objpoints))
