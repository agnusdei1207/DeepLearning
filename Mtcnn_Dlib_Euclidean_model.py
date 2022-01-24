# -*- coding: utf-8 -*-

# MTCNN
!pip3 install facenet
!pip3 install MTCNN
!pip3 install facenet_pytorch
!pip3 install opencv-python
!pip3 install MMCV
!pip3 install IPython
!pip3 install Ipython display
# Dlib 
!pip3 install dlib 
!pip3 install imutils
!pip3 install cmake
!pip3 install scipy

# MTCNN 
from facenet_pytorch import MTCNN 
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython.display  import display
from torchvision import datasets, transforms
from tensorflow import keras
import joblib
#HTML Video
from IPython.display import HTML
from base64 import b64encode
# Dlib
import dlib, imutils
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt
# Euclidean distance
from scipy.spatial import distance
from math import sqrt
import pandas as pd

# MTCNN
def MTCNN_face_detection_extraction(v_path):
  cutted_face_location = [] 
  frame_list = []
  detected_rectangle_faces = []

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  mtcnn = MTCNN(keep_all=True, device=device)
  video = mmcv.VideoReader(v_path)
  frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
  for i, frame in enumerate(frames):
    boxes, _ = mtcnn.detect(frame)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    try:
      for box in boxes:
        if box is not None :
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=5) 
            frame_list.append(frame_draw)
            cutted_face_location.append(box)
    except:
      print("error")
  for i in range(len(frame_list)):
    x,y,w,h=cutted_face_location[i]
    croppedImage=frame_list[i].crop((x,y,w,h))
    detected_rectangle_faces.append(croppedImage)
    display(croppedImage)

  return detected_rectangle_faces

# Dlib
def DLIB_face_landmarks_detection(croppedImage):

  detected_rectangle_faces = []
  nparray_detected_rectangle_faces = []
  coordinate_list = np.zeros(68)

  try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/content/drive/MyDrive/Colab Notebooks/shape_predictor_68_face_landmarks.dat")
  except:
    print("Dlib Predictor read error")

  numpy_image = np.array(croppedImage)
  nparray_detected_rectangle_faces.append(numpy_image)

  for j in nparray_detected_rectangle_faces:
    image = cv2.cvtColor(j, cv2.COLOR_RGB2BGR)
    imgae = cv2.resize(image,(480,480),interpolation = cv2.INTER_CUBIC)
    image = imutils.resize(image, width=500)

    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rectangle = detector(color_image, 1)

    for(i, rect) in enumerate(rectangle):
      LAND = predictor(color_image, rect)
      LAND = face_utils.shape_to_np(LAND)
      print(f"LAND : {LAND}") # (68,2) 68개 2세트

      (x,y,w,h) = face_utils.rect_to_bb(rect)
      cv2.rectangle(color_image, (x,y), (x+w, y+h), (0,255,255), 1)
      cv2.putText(color_image, "Face #{}".format(i+1), (x-1, y-1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,255), 1)

    for(x,y) in LAND:
      cv2.circle(image,(x,y),3,(0,0,255),-1)
      coordinate_list = (x,y)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)

  return LAND

# Euclidean Distance
def calculate_distance(vector_dim):
  dist = []
  for i in range(68):
    if i != 30:
      dist.append(distance.euclidean(vector_dim[30],vector_dim[i]))
  return dist

mtcnn_list = MTCNN_face_detection_extraction("/content/drive/MyDrive/Colab Notebooks/test_park.mp4")
for i in mtcnn_list:
  D = DLIB_face_landmarks_detection(i) # (68,2)
len(calculate_distance(D))

dim = frame_list[0].size # (224, 224)
fourcc = cv2.VideoWriter_fourcc(*'FMP4')  
video_tracked = cv2.VideoWriter("/content/drive/MyDrive/Colab Notebooks/result.mp4", fourcc, 25.0, dim) 
# outputFile (str) – 저장할 파일 & 경로
# fourcc – Codec type >>  cv2.VideoWriter_fourcc()
# frame (float) – 초당 저장될 frame
# size (list) – 저장될 사이즈(ex; 640, 480)
for frame in frame_list:
   video_tracked.write(cv2.cvtColor(np.array(frame), 4)) # np 배열, cv2.COLOR_RGB2BGR = 4 색변환
video_tracked.release() # 열려있는 비디오 닫기 close == release

# 모델 pickle 파일로 저장
joblib.dump(mtcnn, './model_Park.pkl')

# MTCNN 
def detecting_face(v_path):
  cutted_face_location = [] 
  frame_list = []
  detected_rectangle_faces = []

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  mtcnn = MTCNN(keep_all=True, device=device)
  video = mmcv.VideoReader(v_path)
  frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
  for i, frame in enumerate(frames):
    boxes, _ = mtcnn.detect(frame)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    try:
      for box in boxes:
        if box is not None :
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=5) 
            frame_list.append(frame_draw)
            cutted_face_location.append(box)
    except:
      print("error")
  for i in range(len(frame_list)):
    x,y,w,h=cutted_face_location[i]
    croppedImage=frame_list[i].crop((x,y,w,h))
    detected_rectangle_faces.append(croppedImage)
    display(croppedImage)

# Dlib
  nparray_detected_rectangle_faces = []
  coordinate_list = np.zeros(68)

  try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/content/drive/MyDrive/Colab Notebooks/shape_predictor_68_face_landmarks.dat")
  except:
    print("Dlib Predictor read error")

  numpy_image = np.array(croppedImage)
  nparray_detected_rectangle_faces.append(numpy_image)

  for j in nparray_detected_rectangle_faces:
    image = cv2.cvtColor(j, cv2.COLOR_RGB2BGR)
    imgae = cv2.resize(image,(480,480),interpolation = cv2.INTER_CUBIC)
    image = imutils.resize(image, width=500)

    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rectangle = detector(color_image, 1)

    for(i, rect) in enumerate(rectangle):
      LAND = predictor(color_image, rect)
      LAND = face_utils.shape_to_np(LAND)
      print(f"LAND : {LAND}") # (68,2) 68개 2세트

      (x,y,w,h) = face_utils.rect_to_bb(rect)
      cv2.rectangle(color_image, (x,y), (x+w, y+h), (0,255,255), 1)
      cv2.putText(color_image, "Face #{}".format(i+1), (x-1, y-1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,255), 1)

    for(x,y) in LAND:
      cv2.circle(image,(x,y),3,(0,0,255),-1)
      coordinate_list = (x,y)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
# Euclidean
  dist = []
  for i in range(68):
    if i != 30:
      dist.append(distance.euclidean(LAND[30],LAND[i]))
  return dist
