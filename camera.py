import cv2
from keras.preprocessing import image

from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            # # Compute the arithmetic mean.
            roi = image.image_utils.img_to_array(roi)
            roi -= np.mean(roi, axis=0)
            # Compute the standard deviation.
            roi /= np.std(roi, axis=0)
            pred, a = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            best = (max(a[0])) * 100
            best = round(best)
            best = str(int(best))
            cv2.putText(fr, pred + ": " + best + "%", (x, y - 5), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (179, 83, 121), 4)
            EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            offset = 35
            for idx, lbl in enumerate(EMOTIONS_LIST):
                pre = (a[0][idx]) * 100
                pre = round(pre)
                pre = str(int(pre))
                cv2.putText(fr, str(lbl) + ": " + pre + "%", (x + h, y + 20 + offset * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 225, 225), 2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
