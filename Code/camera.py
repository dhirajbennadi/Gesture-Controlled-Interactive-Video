import cv2
import mediapipe as mp
import time
import syslog
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


ioTimerStart = 0
ioTimerEnd = 0
elapsedioTime = 0
firstTime = False
# For webcam input:
cap = cv2.VideoCapture(0)
if cap is not None:
  ioTimerStart = time.time()
cap.set(3,480)
cap.set(4,360)

print("Input Video FPS = {}".format(cap.get(cv2.CAP_PROP_FPS)))
image_width=480
image_height=360

frameCount = 0
startTime = 0
endTime = 0
elapsedDuration = 0
count = 0
zoom_out_cnt = 0
zoom_in_cnt = 0
ActionCount = 0
NoActionCount = 0
NoframeCount = 0
PictureCount = 0
CountBeforeResults = 0
CountLandmarks = 0
multipleHands = 0

flag = 1
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:
    #while cap.isOpened():
    while count < 400:
      #print(count)
      #print(cap.get(cv2.CAP_PROP_FPS))
      success, image = cap.read()
      if not success:
        NoframeCount += 1
        if NoframeCount > 20:
          print("Frames Per Sec = {}".format(1 / elapsedDuration))
          print("IO Time = {}".format(elapsedioTime))
          break
        #print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      startTime = time.time()
      if firstTime is False:
        firstTime = True
        ioTimerEnd = time.time()
        elapsedioTime = ioTimerEnd - ioTimerStart
      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      CountBeforeResults += 1
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) > 1:
          multipleHands += 1
        for hand_landmarks in results.multi_hand_landmarks:
          # print(
          #         f'Index finger tip coordinates: (',
          #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
          #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width})'
          #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height})'
          #     )

          Index_fingerX=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
          Thumb_fingerX=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
          Middle_fingerX=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
          Middle_fingerY=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
          Thumb_fingerY=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height

          #print("Index and middle finger {}".format(abs(Thumb_fingerX - Middle_fingerY)))

          #print('fingertipX: '+str(fingertipX))
          #print(mp_hands.HandLandmark.INDEX_FINGER_TIP)
          rows, cols, _channels = map(int, image.shape)
          dim1 = (640,480)
          dim2 = (320,240)
          # print(abs(Index_fingerX - Thumb_fingerX))
          if ((abs(Index_fingerX - Middle_fingerX)) < 20 and (abs(Index_fingerX - Middle_fingerX)) > 10):
            print("Take picture")
            PictureCount += 1
            cv2.imwrite(f'frames/Image_{time.time()}.jpg',image)
          
          elif ((abs(Index_fingerX - Middle_fingerX)) > 40 and (abs(Thumb_fingerX - Middle_fingerY) < 10)):
            print("Close window")
            cap.release()

          elif (abs(Index_fingerX - Thumb_fingerX)) > 10 :
            #syslog.syslog("count = {} Zoom Out".format(count))
            #print("Zoom out")
            zoom_out_cnt+=1
            #image = cv2.pyrUp(image, dstsize=(480, 640))
            image = cv2.resize(image, dim1,interpolation = cv2.INTER_AREA)
          elif (abs(Index_fingerX - Thumb_fingerX)) <= 10:
            #syslog.syslog("count = {} Zoom In".format(count))
            #print("Zoom in")
            zoom_in_cnt += 1
            #image = cv2.pyrDown(image, dstsize=(240,320))
            image = cv2.resize(image, dim2,interpolation = cv2.INTER_AREA)
          else:
            #print("None")
            NoActionCount += 1
            #syslog.syslog("count = {} Not detected".format(count))

          mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          count+=1
          cv2.imshow('MediaPipe Hands', image)
      #cv2.imwrite(str(count) + ".jpg" , image)



      if cv2.waitKey(1) & 0xFF == 27:
        break
      
      endTime = time.time()
      elapsedDuration += endTime - startTime

      #print("Frames Per Sec = {}".format(1 / elapsedDuration))

syslog.syslog("Zoom Out Count = {}".format(zoom_out_cnt))
syslog.syslog("Zoom In Count = {}".format(zoom_in_cnt))
syslog.syslog("No Action Count = {}".format(NoActionCount))
syslog.syslog("Total Frame = {}".format(count))
syslog.syslog("Picture Count  = {}".format(PictureCount))

print("Zoom Out Count = {}".format(zoom_out_cnt))
print("Zoom In Count = {}".format(zoom_in_cnt))
print("No Action Count = {}".format(NoActionCount))
print("Total Frame = {}".format(count))
print("Picture Count  = {}".format(PictureCount))
print("False Identification = {}".format(multipleHands))
print("Frames Per Sec = {}".format(count / elapsedDuration))

cap.release()
