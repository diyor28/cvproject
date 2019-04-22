import cv2
import time
import dlib
import face_recognition
import numpy as np
from cvision import settings
from imageai.Detection import ObjectDetection
from scipy.spatial import distance as dist
from collections import OrderedDict


class BaseRecognition:
    def __init__(self):
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(model_path=settings.YOLO_PATH)
        self.detector.loadModel()
        self.custom_objects = self.detector.CustomObjects(person=True)

    def detect_objects(self, image):
        detection = self.detector.detectCustomObjectsFromImage(input_image=image, custom_objects=self.custom_objects,
                                                               input_type='array', output_type='array',
                                                               extract_detected_objects=True)

        if detection[1]:
            return [ObjectDetected(image=image, coordinates=points["box_points"],
                                   body=True, face=False)
                    for points, image in zip(detection[1], detection[2])]

    @staticmethod
    def crop(image, points, margin=0, size=None):

        x, y, w, h = points

        x -= margin
        y -= margin
        w += margin
        h += margin

        x, y = max(0, x), max(0, y)
        w, h = max(0, w), max(0, h)

        cropped = image[x:w, y:h]

        if size:
            return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

        if not(all(cropped.shape)):
            return image, points

        return cropped, (x, y, w, h)

    def detect_face(self, body, margin=0):
        rgb = cv2.cvtColor(body.image.copy(), cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb, model='hog')

        if not faces:
            return [body]

        result = []
        for face_coordinates in faces:
            face_coordinates = self.css2xy([face_coordinates])[0]
            cropped_face, face_coordinates = self.crop(body.image, face_coordinates, margin=margin)
            face_object = ObjectDetected(image=cropped_face, coordinates=face_coordinates,
                                         face=True, body=False)
            face_object.change_reference(body.x, body.y)
            result.append(face_object)
        return result

    @staticmethod
    def css2xy(points):
        for i in range(len(points)):
            x, y, w, h = points[i]
            points[i] = (w, x, y, h)
        return points

    @staticmethod
    def xy2css(points):
        for i in range(len(points)):
            w, x, y, h = points[i]
            points[i] = (x, y, w, h)
        return points



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
        self.trackers = []
        self.ct = CentroidTracker(max_disappeared=40, max_distance=60)
        self.total_down = 0
        self.total_frames = 0
        self.total_up = 0

    def reset(self):
        self.total_down = 0
        self.total_frames = 0
        self.total_up = 0

    def track_objects(self, frame_object, bodies):
        rects = []

        if self.total_frames % self.skip_frames == 0:
            self.trackers = []

            for body in bodies:
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(body.x, body.y, body.w, body.h)
                tracker.start_track(frame_object.image, rect)
                self.trackers.append(tracker)
        else:
            for tracker in self.trackers:
                tracker.update(frame_object.image)
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
                # y = [c[1] for c in to.centroids]
                to.centroids.append(centroid)

                if not to.counted:
                    self.total += 1
                    to.counted = True

            self.trackable_objects[objectID] = to

            x, y = centroid

            text = "ID {}".format(objectID)
            cv2.putText(frame_object.image, text, (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame_object.image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            cv2.circle(frame_object.image, (x, y), 4, (0, 255, 0), -1)

        self.total_frames += 1
        return frame_object


class ObjectDetected:
    def __init__(self, image, coordinates=(0, 0, -1, -1), face=False, body=False, age=None, gender=None):
        self.image = image
        self.face = face
        self.body = body
        self.size = image.shape[:2]
        self.coordinates = self.construct(coordinates, self.size)
        self.x, self.y, self.w, self.h = self.coordinates
        self.centroid = ((self.x + self.w)//2, (self.y + self.h)//2)
        self.age = age
        self.gender = gender
        self.full_name = None

    def change_reference(self, x, y):
        x_0 = x + self.x
        y_0 = y + self.y
        w_0 = x + self.w
        h_0 = y + self.h
        self.__init__(self.image, coordinates=(x_0, y_0, w_0, h_0), face=self.face)

    @staticmethod
    def construct(coordinates, size):
        x, y, w, h = coordinates

        if w == -1:
            w = size[0]
        elif h == -1:
            h = size[1]

        return x, y, w, h

    def resize(self, size, return_image=False):
        if type(size) == int:
            size = (size, size)

        image = cv2.resize(self.image.copy(), dsize=size)
        if return_image:
            return image
        self.__init__(image, face=self.face)

    def copy(self, image=None, coordinates=None, face=None):
        image = image or self.image
        coordinates = coordinates or self.coordinates
        face = face or self.face
        return ObjectDetected(image=image, coordinates=coordinates, face=face)


def timeit(method):
    def timed(*args, **kwargs):
        start = time.time()
        res = method(*args, **kwargs)
        runtime = int((time.time() - start)*1000)
        print(f"Runtime: {runtime} ms")
        return res
    return timed
