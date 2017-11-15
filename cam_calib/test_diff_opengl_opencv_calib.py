import numpy as np
import cv2
from math import cos, sin, degrees, radians
from guillou_calib import GuillouCalibrationEngine

def main():

    width = 640.0
    height = 480.0
    fx = 580.0
    fy = 580.0
    cx = width/2.0
    cy = height/2.0

    K = np.mat([[fx, 0., cx],
                [0., fy, cy],
                [0., 0., 1.]])

    img1 = np.zeros((height,width,3), dtype=np.uint8)
    img2 = np.zeros((height,width,3), dtype=np.uint8)
    img3 = np.zeros((height,width,3), dtype=np.uint8)
    img3 += 128



    rx = radians(-60.0)
    ry = radians(10.0)
    rz = radians(+15.0)

    sin_rx = sin(rx)
    cos_rx = cos(rx)
    sin_ry = sin(ry)
    cos_ry = cos(ry)
    sin_rz = sin(rz)
    cos_rz = cos(rz)

    Rx = np.mat([[1.0, 0.0, 0.0],
                [0.0, cos_rx, -sin_rx],
                [0.0, sin_rx, cos_rx]])

    Ry = np.mat([[cos_ry, 0.0, sin_ry],
                [0.0, 1.0, 0.0],
                [-sin_ry, 0.0, cos_ry]])

    Rz = np.mat([[cos_rz, -sin_rz, 0.0],
                [sin_rz, cos_rz, 0.0],
                [0.0, 0.0, 1.0]])

    R = Rx * Ry * Rz

    rv, j = cv2.Rodrigues(R)
    rv = np.mat(rv)
    tv = np.mat([0.0, 0.0, 20.]).T 
    

    obj_pts = np.indices((5,7)).T.reshape(-1,2) - np.r_[2.0, 3.0]
    obj_pts = np.hstack((obj_pts, np.zeros((5*7,1)))) #* 1.0/8.0

    img_pts,_ = cv2.projectPoints(obj_pts, rv, tv, K, None)
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)

    

    cv2.drawChessboardCorners(img1, (5,7), img_pts, True)

    axes_obj = np.array([[0., 0., 0.],
                         [5., 0., 0.],
                         [0., 5., 0.],
                         [0., 0., 5.]], dtype=np.float32) 
    axes_img,_ = cv2.projectPoints(axes_obj, rv, tv, K, None)
    axes_img = axes_img.reshape(-1, 2).astype(np.int32)

    o = tuple(axes_img[0,:])
    ox = tuple(axes_img[1,:])
    oy = tuple(axes_img[2,:])
    oz = tuple(axes_img[3,:])


    cv2.line(img1, tuple(axes_img[0,:]), tuple(axes_img[1,:]), cv2.cv.CV_RGB(255,0,0), 2)
    cv2.line(img1, tuple(axes_img[0,:]), tuple(axes_img[2,:]), cv2.cv.CV_RGB(0,255,0), 2)
    cv2.line(img1, tuple(axes_img[0,:]), tuple(axes_img[3,:]), cv2.cv.CV_RGB(0,0,255), 2)
    
    cv2.rectangle(img1, tuple(x+y for x, y in zip(o,(-5,-5))), tuple(x+y for x, y in zip(o,(5,5))), cv2.cv.CV_RGB(255,255,255))
    cv2.rectangle(img1, tuple(x+y for x, y in zip(ox,(-5,-5))), tuple(x+y for x, y in zip(ox,(5,5))), cv2.cv.CV_RGB(255,0,0))
    cv2.rectangle(img1, tuple(x+y for x, y in zip(oy,(-5,-5))), tuple(x+y for x, y in zip(oy,(5,5))), cv2.cv.CV_RGB(0,255,0))
    cv2.rectangle(img1, tuple(x+y for x, y in zip(oz,(-5,-5))), tuple(x+y for x, y in zip(oz,(5,5))), cv2.cv.CV_RGB(0,0,255))

    
    img_pts = img_pts.reshape(-1, 2).astype(np.int32)
    a = tuple(img_pts[0,:])
    b = tuple(img_pts[4,:])
    c = tuple(img_pts[-5,:])
    d = tuple(img_pts[-1,:])

    A = tuple(obj_pts[0,:])
    B = tuple(obj_pts[4,:])
    C = tuple(obj_pts[-5,:])
    D = tuple(obj_pts[-1,:])

    cv2.rectangle(img1, tuple(x+y for x, y in zip(a,(-5,-5))), tuple(x+y for x, y in zip(a,(5,5))), cv2.cv.CV_RGB(255,128,0))
    cv2.rectangle(img1, tuple(x+y for x, y in zip(b,(-5,-5))), tuple(x+y for x, y in zip(b,(5,5))), cv2.cv.CV_RGB(255,128,0))
    cv2.rectangle(img1, tuple(x+y for x, y in zip(c,(-5,-5))), tuple(x+y for x, y in zip(c,(5,5))), cv2.cv.CV_RGB(255,128,0))
    cv2.rectangle(img1, tuple(x+y for x, y in zip(d,(-5,-5))), tuple(x+y for x, y in zip(d,(5,5))), cv2.cv.CV_RGB(255,128,0))

    #cv2.imshow('img1', img1)
    #cv2.waitKey()

    floor_obj = np.array([[-5., -5., 0.],
                         [5., -5., 0.],
                         [5., 5., 0.],
                         [-5., 5., 0.]], dtype=np.float32)
    floor_img,_ = cv2.projectPoints(floor_obj, rv, tv, K, None)
    floor_img = floor_img.reshape(-1, 2).astype(np.int32)

    a = tuple(floor_img[0,:])
    b = tuple(floor_img[1,:])
    c = tuple(floor_img[2,:])
    d = tuple(floor_img[3,:])

    cv2.line(img3, a, b, cv2.cv.CV_RGB(0,0,0), 3)
    cv2.line(img3, b, c, cv2.cv.CV_RGB(0,0,0), 3)
    cv2.line(img3, c, d, cv2.cv.CV_RGB(0,0,0), 3)
    cv2.line(img3, d, a, cv2.cv.CV_RGB(0,0,0), 3)
     
    cv2.line(img3, o, ox, cv2.cv.CV_RGB(255,0,0), 3)
    cv2.line(img3, o, oy, cv2.cv.CV_RGB(0,255,0), 3)
    cv2.line(img3, o, oz, cv2.cv.CV_RGB(0,0,255), 3)


    a = tuple(img_pts[0,:])
    b = tuple(img_pts[4,:])
    c = tuple(img_pts[-5,:])
    d = tuple(img_pts[-1,:])

    calib_eng = GuillouCalibrationEngine()
    calib_eng.width = width
    calib_eng.height = height
    calib_eng.cx = width/2.0
    calib_eng.cy = height/2.0
    calib_eng.vanishing_lines_x = [a+b, c+d]
    calib_eng.vanishing_lines_y = [a+c, b+d]

    calib_eng.a = a
    calib_eng.b = b

    calib_eng.A = A
    calib_eng.B = B

    calib_eng.calibrate()

    K2 = np.mat([[calib_eng.fx, 0., calib_eng.cx],
                [0., calib_eng.fy, calib_eng.cy],
                [0., 0., 1.]])

    
    R2 = np.mat(calib_eng.R)
    rv2,_ = cv2.Rodrigues(R2)

    tv2 = np.mat(calib_eng.t).T 

    M = np.row_stack((np.column_stack((R2, tv2)),[0.,0.,0.,1.]))
    M = np.mat([[ 1., 0., 0., 0.],
                [ 0., 1., 0., 0.],
                [ 0., 0., 1., 0.],
                [ 0., 0., 0., 1.]]) * M

    R2 = M[:3,:3]
    tv2 = M[:3,3]
    rv2,_ = cv2.Rodrigues(R2)

    img_pts2,_ = cv2.projectPoints(obj_pts, rv2, tv2, K2, None)
    img_pts2 = img_pts2.reshape(-1, 2).astype(np.float32)

    axes_img2,_ = cv2.projectPoints(axes_obj, rv2, tv2, K2, None)
    axes_img2 = axes_img2.reshape(-1, 2).astype(np.int32)


    cv2.drawChessboardCorners(img2, (5,7), img_pts2, True)


    o = tuple(axes_img2[0,:])
    ox = tuple(axes_img2[1,:])
    oy = tuple(axes_img2[2,:])
    oz = tuple(axes_img2[3,:])


    cv2.line(img2, o, ox, cv2.cv.CV_RGB(255,0,0))
    cv2.line(img2, o, oy, cv2.cv.CV_RGB(0,255,0))
    cv2.line(img2, o, oz, cv2.cv.CV_RGB(0,0,255))
    
    cv2.rectangle(img2, tuple(x+y for x, y in zip(o,(-5,-5))), tuple(x+y for x, y in zip(o,(5,5))), cv2.cv.CV_RGB(255,255,255))
    cv2.rectangle(img2, tuple(x+y for x, y in zip(ox,(-5,-5))), tuple(x+y for x, y in zip(ox,(5,5))), cv2.cv.CV_RGB(255,0,0))
    cv2.rectangle(img2, tuple(x+y for x, y in zip(oy,(-5,-5))), tuple(x+y for x, y in zip(oy,(5,5))), cv2.cv.CV_RGB(0,255,0))
    cv2.rectangle(img2, tuple(x+y for x, y in zip(oz,(-5,-5))), tuple(x+y for x, y in zip(oz,(5,5))), cv2.cv.CV_RGB(0,0,255))

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    #cv2.imshow('img3', img3)
    cv2.waitKey()
    cv2.destroyAllWindows()


    #cv2.imwrite('test_img.png', img3)

    return

if __name__ == '__main__':
    main()