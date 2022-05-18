import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

timeMain = time.time()
time1 = 0
time2 = 0
Activity = 0

a = int(input("Time end?: "))
#a = 5.000000000000000
b = 0

if Activity == 1:
    time1 = time.time()
elif Activity == 2:
    time2 = time.time()


Poses = mp_pose.Pose(
    min_detection_confidence=1.0,
    min_tracking_confidence=0.5)


knn = cv2.ml.KNearest_create()

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
      #static_image_mode: 비디오 스트림으로 처리 or not
    model_complexity=2,
      #model_complexity: 모델의 복잡성, 숫자가 올라갈수록 지연시간 증가
    enable_segmentation=True,
      #enable_segmentation: 포즈 말고도 다른 분할 마스크 생성
    min_detection_confidence=1.0) as pose:
      #min_detection_confidence: 탐지성공으로 간주되는 최소 신뢰값. [0.0 < x < 1.0]
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    
    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = mp.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = mp.zeros(image.shape, dtype=mp.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = mp.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)


with mp_pose.Pose(
    min_detection_confidence=0.5,
      #min_detection_confidence: 탐지성공으로 간주되는 최소 신뢰값. [0.0 < x < 1.0]
    min_tracking_confidence=0.5) as pose:
      #min_tracking_confidence: 추적검출 정확도. [0.0 < x < INF]


  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)




    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    

    

    
    try:
      Activity = 1

      mouth_l = pose.process(image).pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
      mouth_R = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
      print()
      print("Activity On")
      print("ActiveTime is ", (time.time() - time1) )
      print("StopTime is ", (time.time() - time2) )
      print()
      print( "입 x:", (mouth_l.x + mouth_R.x)/2 )
      print( "입 y:", (mouth_l.y + mouth_R.y)/2 )
      print( "입 z:", (mouth_l.z + mouth_R.z)/2 )
      print()

      font                   = cv2.FONT_HERSHEY_SIMPLEX
      bottomLeftCornerOfText = (10,500)
      fontScale              = 5
      fontColor              = (255,255,255)
      thickness              = 10
      lineType               = 10
      


      cv2.putText(image, 'mouth detected',
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

      
      
    except AttributeError:
      Activity = 2

      print()
      print("Activity Offed")
      print("ActiveTime is ", (time.time() - time1) )
      print("StopTime is ", (time.time() - time2) )
      print()
      print( "입 x:", "X" )
      print( "입 y:", "X" )
      print( "입 z:", "X" )
      print()
      
      

      if int((time.time() - time2)) >= 3:
        break 
      #time.sleep(1)
    
    
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    
    # Flip the image horizontally for a selfie-view display.
    mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow("img",image)
    
    if cv2.waitKey(1) == ord('q'):
      print("Process End")
      break
    
    if b >= 3:
      print("경고 3회 누적으로 인한 종료, " + str(int((time.time() - time1))) + "시간동안 집중하며, " + str(a) + "회 자리비움.")
      break
    if (time.time() - time1) >= a:
      #print("정상적인 종료, 지정된 " + str(int(((time.time() - time1))/60)) + "시간동안 집중하며, " + str(int((time.time() - time2))) + "초간 자리비움.")
      #print(a, "sec end")
      continue
    
cap.release()