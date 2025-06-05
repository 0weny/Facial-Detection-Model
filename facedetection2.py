import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import urllib.request
from collections import deque

# 모델 경로 및 입력 이미지 크기
model_path = 'emotion_model_2.h5'
shape_x, shape_y = 75, 75
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Surprise', 'Neutral']

# 감정 smoothing을 위한 최근 N개의 값 저장
emotion_buffer = deque(maxlen=5)
frame_count = 0

# 1. 감정 분류 모델 로드
if os.path.exists(model_path):
    model = load_model(model_path)
    print("\u2705 감정 분류 모델 로드 완료")
else:
    print(f"❌ 모델이 존재하지 않습니다: {model_path}")
    exit()

# 2. DNN 얼굴 인식 모델 다운로드 및 로드
proto_path = "deploy.prototxt"
model_file = "res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(proto_path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        proto_path
    )

if not os.path.exists(model_file):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        model_file
    )

face_net = cv2.dnn.readNetFromCaffe(proto_path, model_file)

# 3. 웹캠 실행
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), False, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (shape_x, shape_y))
        except:
            continue

        if frame_count % 3 == 0:
            normalized = resized.astype(np.float32) / 255.0
            input_tensor = np.reshape(normalized, (1, shape_x, shape_y, 1))
            preds = model.predict(input_tensor, verbose=0)[0]
            merged_preds = np.zeros(6)
            merged_preds[0:4] = preds[0:4]
            merged_preds[4] = preds[5]
            merged_preds[5] = preds[4] + preds[6]
            emotion_buffer.append(merged_preds)

        if len(emotion_buffer) > 0:
            avg_preds = np.mean(emotion_buffer, axis=0)
            pred_label = np.argmax(avg_preds)
            label_text = f"{emotions[pred_label]}: {round(avg_preds[pred_label], 2)}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    frame_count += 1
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("DNN Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
