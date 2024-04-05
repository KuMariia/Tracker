import torch
import cv2
import numpy as np
from ultralytics import YOLO

from deep_sort.deep_sort.tracker import Tracker as DSTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections
from deep_sort.application_util.visualization import create_unique_color_uchar


class Tracker:

    def __init__(self, max_cosine_distance=0.3, nn_budget=None):
        self.model = YOLO('yolov8n.pt')
        DS_model_path = 'deep_sort/network/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DSTracker(metric)
        self.encoder = generate_detections.create_box_encoder(DS_model_path, batch_size=1)

        self.single_tracker = cv2.TrackerMIL.create()

        self.device = '0' if torch.cuda.is_available() else 'cpu'

    def get_detections(self, results, frame, conf_threshold=0.4):
        ''' Processes detections from YOLO and converts it to Detection class object used by DeepSort
        Parameters:
            results - objects detected by YOLO
            frame - current frame from webcam
            conf_threshold - threshold of confidence value
        Returns:
            detections - objects detected by YOLO converted to Detection class object
            cls_names - names of objects corresponding to detections'''
        class_names = results[0].names
        boxes = []
        conf = []
        cls_name = []
        for r in results[0].boxes:
            cls = int(r.cls)
            conf_score = float(r.conf.cpu())
            x1, y1, x2, y2 = np.asarray(r.xyxy[0].cpu(), dtype=int)
            w = x2 - x1
            h = y2 - y1
            if conf_score > conf_threshold:
                boxes.append([x1, y1, w, h])
                conf.append(conf_score)
                cls_name.append(class_names[cls])

        features = self.encoder(frame, boxes)
        detections = []
        for i, box in enumerate(boxes):
            detections.append(Detection(box, conf[i], features[i]))

        return detections, cls_name

    def tracking(self, frame):
        ''' Process current frame through YOLO and DeepSort
        Paramerets:
            frame - current frame from webcam
        Returns:
            self.tracker.tracks - objects processed by DeepSort
            cls_names - names of objects corresponding to tracks'''
        result = self.model(frame, device=self.device, verbose=False)

        detections, cls_names = self.get_detections(result, frame)
        self.tracker.predict()
        self.tracker.update(detections)

        return self.tracker.tracks, cls_names

    def draw_boxes(self, frame, tracks, cls_names, mode, single_box_cls):
        ''' Draws bboxes from DeepSort and single tracer
        Parameters:
            frame - current frame from webcam stream
            tracks - coordinates on bboxes we want to track
            cls_names - names of tracking objects
            mode - tracking mode: MULTI or SINGLE
            single_box_cls - class ob object we want to track via single tracker
        Returns:
            frame - processed frame '''
        if mode == 'MULTI':
            for track, cls in zip(tracks, cls_names):
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                x1, y1, x2, y2 = [int(i) for i in track.to_tlbr()]
                x_center = x1 + (x2-x1)//2
                y_center = y1 + (y2-y1)//2
                id = track.track_id

                r, g, b = create_unique_color_uchar(id)
                color = (b, g, r)

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
                frame = cv2.circle(frame, (int(x_center), int(y_center)), radius=0, color=color, thickness=3)
                frame = cv2.putText(frame, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    color=color, thickness=2)

        else:
            x1, y1, w, h = tracks
            x2 = x1 + w
            y2 = y1 + h
            x_center = x1 + w // 2
            y_center = y1 + h // 2
            color = (0, 0, 255)

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
            frame = cv2.circle(frame, (int(x_center), int(y_center)), radius=0, color=color, thickness=3)
            frame = cv2.putText(frame, single_box_cls, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                              color=color, thickness=2)

        return frame


