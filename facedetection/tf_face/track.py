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

from keras.models import load_model

from imageai.Detection import ObjectDetection
import face_recognition

from scipy.spatial import distance as dist
from collections import OrderedDict

from cvision import settings

from .utils import ObjectDetected

graph = tf.get_default_graph()


def timeit(method):
    def timed(*args, **kwargs):
        start = time.time()
        res = method(*args, **kwargs)
        runtime = int((time.time() - start)*1000)
        print(f"Runtime: {runtime} ms")
        return res
    return timed


class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.max_disappeared = max_disappeared
        self.maxDistance = max_distance
        self.args = {'confidence':0.6, 'skip_frames':1}

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])

        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), input_centroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(input_centroids[col])

        # return the set of trackable objects
        return self.objects


class ObjectTracker:
    def __init__(self, skip_frames=5):
        self.skip_frames = skip_frames
        self.total = 0
        self.total_frames = 0
        self.trackable_objects = {}
        self.ct = CentroidTracker(max_disappeared=40, max_distance=60)

    def reset(self):
        self.total_down = 0
        self.total_frames = 0
        self.total_up = 0

    def track_objects(self, frame, detections):
        rects = []

        if self.total_frames % self.skip_frames == 0:
            self.trackers = []

            for i in range(len(detections)):
                detection = detections[i]
                # confidence = detection['percentage_probability']

                box = detection['box_points']
                box = list(map(int, box))
                startX, startY, endX, endY = box

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(frame, rect)
                self.trackers.append(tracker)
        else:
            for tracker in self.trackers:
                tracker.update(frame)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        objects = self.ct.update(rects)

        for objectID, centroid in objects.items():
            to = self.trackable_objects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                y = [c[1] for c in to.centroids]
                to.centroids.append(centroid)

                if not to.counted:
                    self.total += 1
                    to.counted = True

            self.trackable_objects[objectID] = to

            x, y = centroid

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (x-10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        self.total_frames += 1
        return frame


class FaceCV(object):
    def __init__(self):
        self.face_size = settings.FACE_SIZE
        self.model = load_model(settings.AGE_MODEL_PATH)
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(model_path=settings.YOLO_PATH)
        self.detector.loadModel()
        self.custom_objects = self.detector.CustomObjects(person=True)
        self.object_tracker = ObjectTracker()
        self.frames_counter = 0
        self.last_data = []

    def detect_objects(self, image):
        detection = self.detector.detectCustomObjectsFromImage(input_image=image, custom_objects=self.custom_objects,
                                                               input_type='array', output_type='array',
                                                               extract_detected_objects=True)

        if detection[1]:
            return [ObjectDetected(image=image, coordinates=points["box_points"],
                                   body=True, face=False)
                    for points, image in zip(detection[1], detection[2])]

    @staticmethod
    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        image = cv2.rectangle(image, (x, y - size[1]), (x + size[0], y),
                                          (255, 0, 0), cv2.FILLED)
        image = cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
        return image

    @staticmethod
    def crop(image, points, margin=0, size=None):

        x, y, w, h = points

        x -= margin
        y -= margin
        w += margin
        h += margin

        if size:
            return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

        image = image[x:w, y:h]
        if not(all(image.shape)):
            print(x, w, y, h)
            print(image.shape)
            raise ValueError("lkfjdsal")

        return image, (x, y, w, h)

    def detect_face(self, body, margin=0):

        rgb = cv2.cvtColor(body.image.copy(), cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb, model='hog')

        if not faces:
            return [body]

        result = []
        for face_coordinates in faces:
            x, y, w, h = face_coordinates
            face_coordinates = (w, x, y, h)
            cropped_face, face_coordinates = self.crop(body.image, face_coordinates, margin=margin)
            face_object = ObjectDetected(image=cropped_face, coordinates=face_coordinates,
                                         face=True, body=False)
            face_object.change_reference(body.x, body.y)
            result.append(face_object)
        return result

    def preprocess_images(self, frame_object, include_body):
        if include_body:
            rgb = cv2.cvtColor(frame_object.image.copy(), cv2.COLOR_BGR2RGB)
            bodies = self.detect_objects(image=rgb)  # initially detected body images
            if not bodies:
                return [frame_object]

            return [self.detect_face(body, margin=10)[0] for body in bodies]
        else:
            faces = self.detect_face(frame_object)
            return faces

    @timeit
    def analyze_frame(self, frame, scale_factor=1.0, include_body=False, skip_frames=5):

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
                    label = "{}, {}".format(face.age, face.gender)
                    cv2.rectangle(frame_object.image, (face.x, face.y), (face.w, face.h), (255, 200, 0), 2)
                    frame_object.image = self.draw_label(frame_object.image, (face.x, face.y), label)
                return frame_object.image

        with graph.as_default():
            faces = self.preprocess_images(frame_object, include_body=include_body)

            if faces is not None:
                for face in faces:
                    if face.face:
                        results = self.model.predict(face.resize(settings.FACE_SIZE, return_image=True)[np.newaxis])
                        gender = "F" if results[0][0][0] > 0.5 else "M"
                        ages = np.arange(0, 101).reshape(101, 1)
                        age = int(results[1].dot(ages).flatten())

                        face.age = age
                        face.gender = gender

                        label = "{}, {}".format(age, gender)
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

