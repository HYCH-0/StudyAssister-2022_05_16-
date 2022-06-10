from distutils.log import warn
import cv2
import mediapipe as mp
import time

from pkg_resources import add_activation_listener
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

seatNum = 0
name = 0
targeTime = 0

from student_information import admin, student_dict, seat_list, n1, n2, n3

try:
  (seatNum, name, targeTime) = admin(student_dict, seat_list, n1, n2, n3)
except:
  seatNum = 1
  name = 'Default'
  targeTime = 999
  print("유효하지 않은 변수 입력, 기본값으로 기입합니다.\n")

timeMain = time.time()
time1 = 0
time2 = 0

Activity = 0
ActBool = 0

tActiviteTime = 0
ttActiviteTime = 0
tStopTime = 0
ttStopTime = 0

StopCount = 0

Saved = 0
tSaved = 0


warnCount = 0
warnCountTemp = 0

warningTime = 10

pause = 0

Poses = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.3)

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
    min_detection_confidence=0.3) as pose:
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
    
    if Activity == 0:
      print("Activity Ready")
      print()
      print("자리비움 5초당 경고 1회로 계산됩니다.")
      ActiviteTime = 0
      StopTime = 0
      time.sleep(1)

    if Activity == 1:
      
      time2 = time.time()

    elif Activity == 2:
      
      time1 = time.time()
    
    ActiviteTime = int((time.time() - time1))
    StopTime = int((time.time() - time2))

    if pause == 1:
      ActiviteTime = 0
      StopTime = 0
    
    
    try:  #감지되었을 경우 실행

      if Activity == 0: #처음 실행시 값 초기화 후 실행
        Activity = 2
      else: #값 초기화 후에 메인 코드 실행
        Activity = 1
        if (ActiviteTime or StopTime) < 1000000000: #일정 값 벗어날 경우 실행불가 
          
          shoulder_l = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
          shoulder_r = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

          if ActiviteTime != 0: #1초 이상의 값을 가질때 tA값에 저장, 저장되었다는 표시
            tActiviteTime = ActiviteTime
            #Saved = 1
            tSaved = 1
          else: #0초일때 이전 값을 
            #Saved = 0
            if tSaved == 1: #모드 변화시 이전 값 저장
              ttActiviteTime += tActiviteTime
              tSaved = 0

          #print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
          print("--------집중모드--------")
          print("집중 시간:", ActiviteTime)
          print("누적 집중 시간:", ActiviteTime + ttActiviteTime )
          print("경고 횟수:", warnCount)
          print()
          print( "어깨 x:", (shoulder_l.x + shoulder_r.x)/2 )
          print( "어깨 y:", (shoulder_l.y + shoulder_r.y)/2 )
          print( "어깨 z:", (shoulder_l.z + shoulder_r.z)/2 )
          #print("Saved:", Saved)
          print()

          bottomLeftCornerOfText = (50,615) 
          font                   = cv2.FONT_HERSHEY_SIMPLEX
          fontScale              = 1
          if warnCount == 1:
            fontColor              = (220,20,60)
          elif warnCount == 2:
            fontColor              = (255,0,0)
          else:
            fontColor              = (255,255,255)
          thickness              = 2
          lineType               = 1
          
          cv2.putText(image, str("ActivateTime: " + str(ActiviteTime) + "  Accumulate Time: " + str(ActiviteTime + ttActiviteTime)),
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            (255,255,255),
            thickness,
            lineType)

          cv2.putText(image, str("Warning Count: " + str(warnCount)),
            (50,500), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
          
          cv2.putText(image, str("SeatNumber: " + str(seatNum) + " Name: " + str(name) + " TargetTime: " + str(targeTime)),
            (50,385), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    except AttributeError: #감지되지 않았을 경우 실행

      if Activity == 0: #처음 실행시 값 초기화 후 실행
        Activity = 1
      else: #값 초기화 후에 메인 코드 실행
        Activity = 2
        if (ActiviteTime or StopTime) < 1000000000: #일정 값 벗어날 경우 실행불가

          if StopTime != 0: #1초 이상의 값을 가질때 tS를 1로저장, 저장되었다는 표시
            tStopTime = StopTime
            #Saved = 1
            tSaved = 1
          
          else: #0초일때 이전 값을 ttS에 저장하며, tS를 0으로 저장
            #Saved = 0
            if tSaved == 1: #모드 변화시 이전 값 저장
              ttStopTime += tStopTime
              tSaved = 0
            
          #print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
          print("--------자리비움--------")
          print("자리비움 시간:", StopTime)
          print("누적 자리비움 시간:", StopTime + ttStopTime )
          print("경고 횟수:", warnCount)
          print()
          print( "어깨 x:", "X" )
          print( "어깨 y:", "X" )
          print( "어깨 z:", "X" )
          #print("Saved:", Saved)
          print()

          bottomLeftCornerOfText = (50,615)
          font                   = cv2.FONT_HERSHEY_SIMPLEX
          fontScale              = 1
          if warnCount == 1:
            fontColor              = (220,20,60)
          elif warnCount == 2:
            fontColor              = (255,0,0)
          else:
            fontColor              = (0,0,0)
          thickness              = 2
          lineType               = 1
          
          cv2.putText(image, str("StopTime: " + str(StopTime) + "  Accumulate Time: " + str(StopTime + ttStopTime)),
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            (0,0,0),
            thickness,
            lineType)

          cv2.putText(image, str("Warning Count: " + str(warnCount)),
            (50,500), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
          
          cv2.putText(image, str("SeatNumber: " + str(seatNum) + " Name: " + str(name) + " TargetTime: " + str(targeTime)),
            (50,385), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
          
          
          if ((StopTime) // warningTime) != warnCountTemp:
            if (StopTime) // warningTime == 0:
              warnCountTemp = 0
            else:
              warnCount += 1
              warnCountTemp = (StopTime) // warningTime
    
    #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    
    # Flip the image horizontally for a selfie-view display.
    mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    cv2.imshow("studyassist",image)
    
    if cv2.waitKey(1) == ord('p'):
      pause = 1
      input("GUI 일시정지됨: 계속하려면 터미널 상에서 Enter키를 눌러주세요")
      pause = 0

    if cv2.waitKey(1) == ord('q'):
      print("강제종료됨. ")
      break

    if (ActiviteTime or StopTime) < 1000000000:
      if warnCount >= 3 or int(ActiviteTime + ttActiviteTime) >= int(targeTime):
        if warnCount >= 3:
          #print("일정시간 자리비움으로 인한 종료, " + str(ActiviteTime + ttActiviteTime) + "초동안 집중하며, " + str(StopTime + ttStopTime) + "초간 자리비움.")
          print(str(seatNum) + "번자리의 " + str(name) + "님 경고 3회 이상으로 인해 종료되었습니다.\n누적 " + str(StopTime + ttStopTime) + "초간, 한번에 " + str(StopTime) + "초동안 자리비움.")
          break
        if int(ActiviteTime + ttActiviteTime) >= int(targeTime):
          #print("정상적인 종료, 지정된 " + str(ActiviteTime + ttActiviteTime) + "초동안 집중하며, " + str(StopTime + ttStopTime) + "초간 자리비움.")
          print(str(seatNum) + "번자리의 " + str(name) + "님 시간이 만료되었습니다.\n누적 " + str(ActiviteTime + ttActiviteTime) + "초간, 한번에 " + str(ActiviteTime) + "초동안 집중함.")
          break
    
cap.release()