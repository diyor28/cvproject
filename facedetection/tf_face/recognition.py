import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import cv2
import dlib
import numpy as np
import time

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# config = tf.ConfigProto()
#
# init_op = tf.global_variables_initializer()
# sess = tf.Session(config=config)
# sess.run(init_op)


import face_recognition

from typing import List
from keras.models import load_model

from cvision import settings
from facedetection.models import EmbeddingsModel
from .utils import (ObjectDetected, timeit,
                    ObjectTracker, BaseRecognition)


graph = tf.get_default_graph()


class RecognizeFace(BaseRecognition):
    def __init__(self):
        super(RecognizeFace, self).__init__()

    def create_embedding(self, frame):
        return list(face_recognition.face_encodings(frame)[0])

    @staticmethod
    def recognize_face(faces: List[ObjectDetected]):
        for face in faces:
            if face.face:
                db_records = np.array(EmbeddingsModel.objects.all())
                db_embeddings = np.array([record.embedding for record in db_records])
                w, h = face.image.shape[:2]
                unknown_encoding = face_recognition.face_encodings(face.image,
                                                                   known_face_locations=RecognizeFace.xy2css([(0, 0, w, h)]))
                res = face_recognition.compare_faces(db_embeddings, unknown_encoding)
                face.full_name = db_records[res][0].full_name
        return faces


class FaceCV(BaseRecognition):
    def __init__(self):
        super(FaceCV, self).__init__()
        self.face_size = settings.FACE_SIZE
        self.model = load_model(settings.AGE_MODEL_PATH)
        self.object_tracker = ObjectTracker(skip_frames=settings.SKIP_FRAMES)
        self.frames_counter = 0
        self.last_data = []

    @staticmethod
    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        image = cv2.rectangle(image, (x, y - size[1]), (x + size[0], y),
                                          (255, 0, 0), cv2.FILLED)
        image = cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
        return image

    def preprocess_images(self, frame_object, include_body):
        if include_body:
            rgb = cv2.cvtColor(frame_object.image.copy(), cv2.COLOR_BGR2RGB)
            bodies = self.detect_objects(image=rgb)  # initially detected body images
            if not bodies:
                return frame_object, bodies

            frame_object = self.object_tracker.track_objects(frame_object, bodies)
            faces = [self.detect_face(body, margin=0)[0] for body in bodies]
            return frame_object, faces
        else:
            faces = self.detect_face(frame_object)
            frame_object = self.object_tracker.track_objects(frame_object, faces)
            return frame_object, faces

    @timeit
    def analyze_frame(self, frame, scale_factor=1.0, include_body=False, include_identity=False, skip_frames=5):

        frame_object = ObjectDetected(image=frame, face=False, body=False)
        width = int(frame_object.image.shape[0] * scale_factor)
        height = int(frame_object.image.shape[1] * scale_factor)
        frame_object.resize((height, width))

        if skip_frames:
            self.frames_counter += 1
            if self.frames_counter % skip_frames == 0:
                if self.last_data is None:
                    return frame_object.image

                for face in self.last_data:
                    if face.age and face.gender:
                        label = "{}, {}".format(face.age, face.gender)

                        if include_identity and face.full_name:
                            label += ', ' + face.full_name

                        frame_object.image = self.draw_label(frame_object.image, (face.x, face.y), label)
                    cv2.rectangle(frame_object.image, (face.x, face.y), (face.w, face.h), (255, 200, 0), 2)
                return frame_object.image

        with graph.as_default():
            frame_object, faces = self.preprocess_images(frame_object, include_body=include_body)

            if faces is None:
                self.last_data = None
                return frame_object.image

            if include_identity:
                faces = RecognizeFace.recognize_face(faces)

            for face in faces:
                if face.face:
                    results = self.model.predict(face.resize(settings.FACE_SIZE, return_image=True)[np.newaxis])
                    gender = "F" if results[0][0][0] > 0.5 else "M"
                    ages = np.arange(0, 101).reshape(101, 1)
                    age = int(results[1].dot(ages).flatten())

                    face.age = age
                    face.gender = gender

                    label = "{}, {}".format(age, gender)
                    if include_identity and face.full_name:
                        label += ', ' + face.full_name
                    self.draw_label(frame_object.image, (face.x, face.y), label)

                if face.face or face.body:
                    frame_object.image = cv2.rectangle(frame_object.image, (face.x, face.y),
                                                       (face.w, face.h), (255, 200, 0), 2)

            self.last_data = faces

            return frame_object.image

    def analyze_stream(self, path=None, include_body=False):
        path = path or 0
        video = cv2.VideoCapture(path)

        while True:
            if not video.isOpened():
                time.sleep(5)

            ret, frame = video.read()

            if not ret:
                break

            frame = self.analyze_frame(frame, include_body=include_body)
            if frame is None:
                continue

            cv2.imshow('Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break

        video.release()
        cv2.destroyAllWindows()


def main():
    face = FaceCV()
    face.analyze_stream()


if __name__ == "__main__":
    main()

