import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from visual import *


import copy
import pickle

video1_capture = cv2.VideoCapture(1)
video1_capture.set(3, 1280)
video1_capture.set(4, 1024)
video1_capture.set(5, 60)
video2_capture = cv2.VideoCapture(2)
video2_capture.set(3, 1280)
video2_capture.set(4, 1024)
video2_capture.set(5, 60)
webcam_capture = cv2.VideoCapture(0)
webcam_capture.set(5, 40)

c1_isflip = False
c2_isflip = True
ret1, frame1 = video1_capture.read()
ret2, frame2 = video1_capture.read()
with open('table/0509_Stereo_cvCalibration_result.pickle', 'rb') as fr:
    result = pickle.load(fr)
csv_output = 'data/0830_half_5_measurement.csv'
with open('csv_output', 'w') as fw:
    fw.write('x,y,z,ts\n')
# print(frame1.shape[::-1])
frame1 =cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2 =cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

mtx1 = np.array([[2441.347367, 0.000000, 661.608922]
                    , [0.000000, 2437.002900, 509.925022]
                    , [0, 0, 1]])
RT1 = np.array([[1,  0,  0, 0], [0,  1,  0, 0], [0, 0, 1, 0]])
dist1 = np.array([-0.418326, 12.064726, 0.006454, 0.006439, 0.0])


mtx2 = np.array([[2412.136458, 0.000000, 742.768353],
                 [0.000000, 2409.743589, 569.840267],
                 [0, 0, 1]])

dist2 = np.array([0.013682, -4.051145, 0.005566, 0.004979, 0.0])

RT2 = np.array([[0.695238, -0.000966, 0.718779, -236.973392],
                [0.012590, 0.999862, -0.010835, -2.722954],
                [-0.718670, 0.016582, 0.695154, 99.387476]])


mtx11, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, frame1.shape[::-1], 1,frame1.shape[::-1])
mtx22, roi2 = cv2.getOptimalNewCameraMatrix(mtx2, dist2, frame2.shape[::-1], 1,frame2.shape[::-1])

x1, y1, w1, h1 = roi1
x2, y2, w2, h2 = roi2


project1 = np.dot(mtx1, RT1)
project2 = np.dot(mtx2, RT2)

fundmat = np.array([[0.000000045798, 0.000002861108, -0.001156182167],
                    [0.000001063306, -0.000000066041, -0.016393629477],
                    [-0.000208264931, 0.014081405328, 1.000000000000]])

rectifyL = np.array([[0.014579554938, 0.001154122909, -7.486695574945],
                     [-0.000158364448, 0.014148620255, 0.219604079591],
                     [-0.000000942871, 0.000000106424, 0.014529286018]])

rectifyR = np.array([[1.115050322790, 0.020998083523, -84.383225349128],
                     [0.073353897045, 1.001558661996, -47.744529051258],
                     [0.000180043107, 0.000003390484, 0.883036483662]])

cameraMatrix1, rotMatrix1, transVect1, rotMatrixX1, rotMatrixY1, rotMatrixZ1, eulerAngles1 = cv2.decomposeProjectionMatrix(project1)
cameraMatrix2, rotMatrix2, transVect2, rotMatrixX2, rotMatrixY2, rotMatrixZ2, eulerAngles2 = cv2.decomposeProjectionMatrix(project2)

R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))



#R1, R2, project1, project2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=mtx1, cameraMatrix2=mtx2, distCoeffs1=dist1, distCoeffs2=dist2, imageSize=frame1.shape[::-1], R=rotMatrix2, T=transVect2)

#left_maps = cv2.initUndistortRectifyMap(mtx1, dist1, rectifyL, project1, (1280, 1024), cv2.CV_16SC2)
#right_maps = cv2.initUndistortRectifyMap(mtx2, dist2, rectifyR, project2, (1280, 1024), cv2.CV_16SC2)

print('mtx1')
print(cameraMatrix1)
print(rotMatrix1)
print(rotMatrixX1)
print(rotMatrixY1)
print(rotMatrixZ1)
print(transVect1)
print(eulerAngles1)
print('mtx2')
print(cameraMatrix2)
print(rotMatrix2)
print(rotMatrixX2)
print(rotMatrixY2)
print(rotMatrixZ2)
print('transvec2', transVect2)
print(eulerAngles2)

print(len(result))
target_shape = (2, 2)
# start tracing object
# start tracing object
fd = open('4d', 'w')

# plotting objects and centroid-tracing
scene1 = display(title = 'Virtual curve',x=5, y=0, width=800, height=600
                 , center=(48.5978, -25.6811, 257.772), background=(0, 0, 0))

zoom_ratio = 1.051745898  # 0.98449379

balls = []
face_balls = []

pointer = []

for i in range(3): # XYZ axis
    ax = [0, 0, 0]
    ax[i]=50
    axiom = arrow(pos = (0,0,0), axis = vector(ax), shaftwidth=zoom_ratio/10)

# lower jaw tracer balls definition 4*(0, 0, 0)
for i in range(4):
    balls.append(sphere(pos=(0, 0, 0), radius=zoom_ratio/0.5, color=color.green))
center = sphere(pos=(0, 0, 0), radius=zoom_ratio/0.5, color=color.red)
move_trace = curve(radius=zoom_ratio/1)

#square = curve(pos=[(0,0),(0,100),(100,100),(100,0),(0,0)], color=color.yellow)

# facebow balls definition  5*(0, 0, 0)
for i in range(5):
    face_balls.append(sphere(pos=(0, 0, 0), radius=zoom_ratio/0.3, color=color.blue))

lower_plane = curve(pos=(0, 0, 0), radius=zoom_ratio/10, color=color.yellow, retain=1)
upper_facebow = curve(pos=(0, 0, 0), radius=zoom_ratio/0.5, color=color.cyan, retain=1)
        # drawing normal vector of tracer
pointer = arrow(pos=(center.pos), axis=vector(0, 0, 0), shaftwidth=3, length=0.5, color=color.red)



fourD = np.zeros((target_shape[0]*target_shape[1], 4), np.float32)
face_fourD = np.zeros((target_shape[0]*target_shape[1], 4), np.float32)
threeD = np.zeros((4, 1, 3), np.float32)


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params2 = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

params2.minThreshold = 10
params2.maxThreshold = 200

# Filtered by Color RGBA
params.filterByColor = True
params.blobColor = 255

params2.filterByColor = True
params2.blobColor = 255
# Filter by Area.
params.filterByArea = True
params.minArea = 100
params.maxArea = 340

params2.filterByArea = True
params2.minArea = 500
params2.maxArea = 4800  #1000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.5

params2.filterByCircularity = False
params2.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.5

params2.filterByConvexity = False
params2.minConvexity = 0.5

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.5

params2.filterByInertia = False
params2.minInertiaRatio = 0.5

start_time = time.time()
i = 0

fw = open(csv_output, 'a')
while True:
    i += 1
    start_time = time.time()
    #rate(100)
    ret1, frame1 = video1_capture.read()
    ret2, frame2 = video2_capture.read()
    ret3, frame3 = webcam_capture.read()

    if ret3:
        cv2.imshow("webcam", frame3)

    # because c1 is upside down
    if c1_isflip:
        frame1 = cv2.flip(frame1, 0)
    if c2_isflip:
        frame2 = cv2.flip(frame2, 0)
    if ret1 and ret2:

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # undistort stereo images
        #gray1 = cv2.undistort(gray1, mtx1, dist1)
        #gray2 = cv2.undistort(gray2, mtx2, dist2)

        #gray1 = gray1[y1:y1+h1, x1:x1+w1]
        #gray2 = gray2[y2:y2+h2, x2:x2+w2]

        #gray1 = cv2.resize(gray1, (1280, 1024), interpolation=cv2.INTER_AREA)
        #gray2 = cv2.resize(gray2, (1280, 1024), interpolation=cv2.INTER_AREA)

# Brightness Adjustment
        phi = 1
        theta = 1
        dark_ratio = 1

        maxIntensity = 255.0  # depends on dtype of image data
        x = arange(maxIntensity)

        gray1 = (maxIntensity/phi) * (gray1/(maxIntensity/theta))**dark_ratio
        gray1 = np.array(gray1, dtype=uint8)

        gray2 = (maxIntensity/phi)*(gray2/(maxIntensity/theta))**dark_ratio
        gray2 = np.array(gray2, dtype=uint8)
        # Brightness Adjustment

        # upper and lower detector definition
        detector1 = cv2.SimpleBlobDetector_create(params)  # tracer blob detector
        detector2 = cv2.SimpleBlobDetector_create(params)

        detector3 = cv2.SimpleBlobDetector_create(params2)  # facebow blob detector
        detector4 = cv2.SimpleBlobDetector_create(params2)

        circles1 = detector1.detect(gray1)  # tracer points
        circles2 = detector2.detect(gray2)

        circles3 = detector3.detect(gray1)  # facebow points
        circles4 = detector4.detect(gray2)


        # Triangulating tracer points
        if len(circles1) == 4 and len(circles2) == 4 and len(circles3) == 5 and len(circles4) == 5:
            #print('circle size')
            #print(circles2[0].size)
            circles1_array = np.array([[circles1[i].pt] for i in range(4)])
            circles2_array = np.array([[circles2[i].pt] for i in range(4)])

            circles3_array = np.array([[circles3[i].pt] for i in range(5)])
            circles4_array = np.array([[circles4[i].pt] for i in range(5)])
            #print('circles1_array')
            #print type(circles1_array)
            #print(circles1_array)

            new_circles1_array = np.reshape(circles1_array, (1, 4, 2 ))
            new_circles2_array = np.reshape(circles2_array, (1, 4, 2))

            new_circles3_array = np.reshape(circles3_array, (1, 5, 2))
            new_circles4_array = np.reshape(circles4_array, (1, 5, 2))
            # n_circles1_array = circles1_array.reshape((1, 4, 2))
            #print('new_circles3', new_circles3_array)
            #print('new_circles4', new_circles4_array)
            #print('new_cir',n_circles1_array)
            #newpt1_array, newpt2_array = cv2.correctMatches(fundmat, new_circles1_array, new_circles2_array)
            #newpt3_array, newpt4_array = cv2.correctMatches(fundmat, new_circles3_array, new_circles4_array)

            #register for sorting all the points


            new_circles1_array[0] = new_circles1_array[0][np.argsort(new_circles1_array[0][:, 0])]
            newpt1_array = new_circles1_array[0].reshape((1, 4, 2))
            new_circles2_array[0] = new_circles2_array[0][np.argsort(new_circles2_array[0][:, 0])]
            newpt2_array = new_circles2_array[0].reshape((1, 4, 2))
            new_circles3_array[0] = new_circles3_array[0][np.argsort(new_circles3_array[0][:, 0])]
            newpt3_array = new_circles3_array[0].reshape((1, 5, 2))
            new_circles4_array[0] = new_circles4_array[0][np.argsort(new_circles4_array[0][:, 0])]
            newpt4_array = new_circles4_array[0].reshape((1, 5, 2))

            """
            temp1Dict = {}
            for x, y in new_circles1_array[0]:
                temp1Dict[x] = y
            newpt1_array = []
            for x in sorted(temp1Dict.keys()):
                newpt1_array.append([x, temp1Dict[x]])
            newpt1_array = np.array([newpt1_array])
            """

            fourD = cv2.triangulatePoints(project1, project2, newpt1_array, newpt2_array)
            face_fourD = cv2.triangulatePoints(project1, project2, newpt3_array, newpt4_array)
            #threeD = cv2.convertPointsFromHomogeneous(fourD)

        gray1 = cv2.drawKeypoints(gray1, circles1, np.array([]), (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        gray2 = cv2.drawKeypoints(gray2, circles2, np.array([]), (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        gray1 = cv2.drawKeypoints(gray1, circles3, np.array([]), (0, 255, 0),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        gray2 = cv2.drawKeypoints(gray2, circles4, np.array([]), (0, 255, 0),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



        #cv2.imshow('left', gray1)
        #cv2.waitKey(5)
        #cv2.imshow('right', gray2)
        #cv2.waitKey(5)
        stereo = np.concatenate((gray1, gray2), axis=1)  # combine image
        stereo2 = cv2.resize(stereo, (1280, 512), interpolation=cv2.INTER_AREA)

        cv2.imshow("monitor", stereo2)
        cv2.waitKey(1)
        cv2.imwrite('combine.jpg', stereo)
        #cv2.imwrite('left_cam.jpg', gray1)
        #cv2.imwrite('right_cam.jpg', gray2)

#    for i in range(len(balls)):
#        balls[i].pos = (fourD[0][i], fourD[1][i], fourD[2][i])

    temp = vector(0, 0, 0)
    face_temp = vector(0, 0, 0)


    # move_max = 10
    # move = vector(random.uniform(-1*move_max,move_max), random.uniform(-1*move_max, move_max), random.uniform(-1*move_max, move_max))
    # print(threeD.shape)
    threeD = threeD.reshape((4, 3))

    # condition loop and drawing balls and line
    if len(circles1) == 4 and len(circles2) == 4 and len(circles3) == 5 and len(circles4) == 5:
        for i in range(4):
            balls[i].pos = (zoom_ratio*(fourD[0][i]/fourD[3][i]), -1*zoom_ratio*(fourD[1][i]/fourD[3][i]), zoom_ratio*(fourD[2][i]/fourD[3][i]))
            # balls[i].pos = ((threeD[i][0]-original_median[0]) * zoom_ratio, (threeD[i][1]-original_median[1]) * zoom_ratio, (threeD[i][2]-original_median[2]) * zoom_ratio)
            temp = balls[i].pos + temp
        for i in range(5):
            # balls[i].pos += move
            face_balls[i].pos = (zoom_ratio*(face_fourD[0][i]/face_fourD[3][i]), -1*zoom_ratio*(face_fourD[1][i]/face_fourD[3][i]), zoom_ratio*(face_fourD[2][i]/face_fourD[3][i]))
            face_temp = face_balls[i].pos + face_temp


        # updated curve positions in drawing loops

        lower_plane.pos = [balls[0].pos, balls[2].pos, balls[3].pos, balls[1].pos, balls[0].pos]
        upper_facebow.pos = [face_balls[0].pos, face_balls[2].pos, face_balls[4].pos, face_balls[3].pos,
                             face_balls[2].pos,
                             face_balls[1].pos, face_balls[0].pos]

        # caculating normal vector
        tracer_vect1 = np.array([balls[0].pos[0]-balls[1].pos[0], balls[0].pos[1]-balls[1].pos[1], balls[0].pos[2]-balls[1].pos[2]])
        tracer_vect2 = np.array([balls[0].pos[0]-balls[2].pos[0], balls[0].pos[1]-balls[2].pos[1], balls[0].pos[2]-balls[2].pos[2]])

        normal_vector = np.cross(tracer_vect1, tracer_vect2)



        # drawing normal vector of tracer
        pointer.pos = center.pos
        pointer.axis = vector(-1*normal_vector)


    center.pos = (temp/4)
    move_trace.append(center.pos)

    # Post-Coordinate Transformation
    Theta = 0
    Phi = 22.972
    Omega = 0

    RotX = np.array([[1, 0, 0],
                     [0, 0.99971562, 0.02384692],
                     [0, -0.02384692, 0.99971562]])
    RotY = np.array([[0.92069, 0, 0.39028],
                     [0, 1, 0],
                     [-0.39028, 0, 0.92069]])
    RotZ = np.array([[-0.99983608, 0, 0],
                     [0, -0.99983608, 0],
                     [0, 0, 1]])
    PostR = np.dot(RotX, RotY, RotZ)

    center.pos = center.pos + np.array([[76], [38.42], [-426.31607]])
    center.pos = np.dot(PostR, center.pos)

    # Post-Coordinate Transformation


    clock = time.time()%86400+28800
    Clock = str(clock)
    print(center.pos)
    #print('fps: %.2f'%(1/(time.time()-start_time)))


    output_string = "%f,%f,%f,%f\n" % (center.pos[0], center.pos[1], center.pos[2], clock)
    fw.write(output_string) # output file define before while-loop


fw.close()
video1_capture.release()
video2_capture.release()
cv2.destroyAllWindows()
