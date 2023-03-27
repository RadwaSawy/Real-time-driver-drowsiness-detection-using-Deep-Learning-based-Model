
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
import time
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# determine the indices of right eye and left eye that are returned from dlib
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# Use dlib to load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model_eye = load_model('vgg16_gray.h5')
model_mouth = load_model('vgg16_yawn.h5')
path = os.getcwd()

video_path = 'Videos\MaleGlasses-5.avi'
cap = cv2.VideoCapture(0)

'''# count the number of frames
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# calculate duration of the video
seconds = int(frames / fps)
video_time = str(datetime.timedelta(seconds=seconds))
print('fps: ', fps)
print("duration in seconds:", seconds)
print("video time:", video_time)'''

thicc = 2
score = 0
score_mouth = 0
r_Eye = None; l_Eye = None; m_mouth = None

flag = 0
total_time = 0
test_duration = 60
begin = time.time()
c_frame = 0


while True:
    ret, frame = cap.read()
    while (ret):

        frame = imutils.resize(frame, width=450)
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)  # use different scale factor
            shape = face_utils.shape_to_np(shape)

            # Left eye detection
            leftEye = shape[lStart:lEnd]
            (x, y, w, h) = cv2.boundingRect(np.array([leftEye]))
            roi_l = frame[y-10:y + h+10, x-10:x + w+10]
            roi_l = cv2.resize(roi_l, (100, 100))
            # cv2.imshow("ROI", roi_l)

            roi_l = np.expand_dims(roi_l, axis=0)
            l_eye = tf.keras.applications.vgg16.preprocess_input(roi_l)
            lpred = model_eye.predict(l_eye)
            l_Eye = lpred[0][0]
            # print('l_Eye', l_Eye)

            # Right eye detection:
            rightEye = shape[rStart:rEnd]
            (x, y, w, h) = cv2.boundingRect(np.array([rightEye]))
            roi_r = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            roi_r = cv2.resize(roi_r, (100, 100))
            # cv2.imshow("ROI", roi_r)

            roi_r = np.expand_dims(roi_r, axis=0)
            r_eye = tf.keras.applications.vgg16.preprocess_input(roi_r)
            rpred = model_eye.predict(r_eye)
            r_Eye = lpred[0][0]
            # print('r_Eye', r_Eye)

            # mouth detection:
            outMouth = shape[mStart:mEnd]
            (x, y, w, h) = cv2.boundingRect(np.array([outMouth]))
            roi_m = frame[y - 12:y + h + 12, x - 10:x + w + 10]
            roi_m = cv2.resize(roi_m, (100, 100))
            cv2.imshow("ROI", roi_m)

            roi_m = np.expand_dims(roi_m, axis=0)
            mou = tf.keras.applications.vgg16.preprocess_input(roi_m)
            mpred = model_eye.predict(mou)
            m_mouth = mpred[0][0]
            print('m_mouth', m_mouth)

            # Predicting:
            if r_Eye != None and l_Eye != None and (r_Eye > 0.8 and l_Eye > 0.8):
                score = score + 1
                cv2.putText(frame, "Closed Eye", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Opened Eye", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if m_mouth != None and m_mouth < 0.2:
                score_mouth = score_mouth + 1
                cv2.putText(frame, "Yawn", (10, height - 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No Yawn", (10, height - 50), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if total_time > test_duration:
                print('time taken : ', total_time)
                begin = time.time()
                if score > 15*(total_time/60) or score_mouth > 2*(total_time/60):
                    print('Number of blinks: ', score)
                    print('Number of Yawns: ', score_mouth)
                    print('15*(total_time/60)', 15*(total_time/60))
                    flag = 1
                    # cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                print("Total number of detected blinks: ", score)
                print("Number of frames during this duration: ", c_frame)
                c_frame = 0
                scoo = score
                score = 0
                total_time = 0
            else:
                print('c_frame', c_frame)
                final = time.time()
                total_time = final - begin
                c_frame += 1

            if flag == 1 and c_frame < 75:
                cv2.putText(frame, 'DOWSINESS ALERT!', (100, height - 300), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Number of closed eye:' + str(scoo), (100, height - 280), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                if (thicc < 16):
                    thicc = thicc + 2
                else:
                    thicc = thicc - 2
                    if (thicc < 2):
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
                try:
                    sound.play()
                except:
                    pass
            else:
                flag = 0

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        ret, frame = cap.read()
    break
cap.release()
cv2.destroyAllWindows()
