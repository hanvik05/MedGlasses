# import time
# import threading
# import argparse
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import pyttsx3
# import queue


# # ---------------- Config ----------------
# MODEL = "yolov8n.pt"   
# PHONE_URL = "http://192.168.1.70:4747/video" 
# TTS_RATE = 150
# MIN_CONF = 0.4
# SPEECH_COOLDOWN = 1.0
# DETECT_EVERY_N_FRAMES = 5   # run YOLO only every 5 frames

# # ---------------- Utilities ----------------
# def bbox_center_x(bbox, frame_w):
#     x1, y1, x2, y2 = bbox
#     cx = (x1 + x2) / 2
#     rel = (cx - frame_w/2) / (frame_w/2)
#     return rel

# def rel_to_clock(rel):
#     if rel < -0.4: return "ten o'clock"
#     if rel < -0.15: return "nine o'clock"
#     if rel < 0.15: return "twelve o'clock"
#     if rel < 0.4: return "three o'clock"
#     return "two o'clock"

# def estimate_distance_from_bbox(bbox, frame_h):
#     x1, y1, x2, y2 = bbox
#     h = abs(y2-y1)
#     frac = h / frame_h
#     if frac < 0.05: return 4.0
#     if frac < 0.12: return 2.0
#     if frac < 0.25: return 1.0
#     return 0.4

# # ---------------- Speech Engine ----------------
# # class Announcer:
# #     def __init__(self, rate=TTS_RATE):
# #         self.engine = pyttsx3.init()
# #         self.engine.setProperty('rate', rate)
# #         self.lock = threading.Lock()
# #         self.last_time = 0

# #     def say(self, text, force=False):
# #         with self.lock:
# #             now = time.time()
# #             if not force and (now - self.last_time) < SPEECH_COOLDOWN:
# #                 return
# #             self.last_time = now
# #             threading.Thread(target=self._speak, args=(text,)).start()

# #     def _speak(self, text):
# #         self.engine.say(text)
# #         self.engine.runAndWait()


# class Announcer:
#     def __init__(self, rate=150):
#         self.engine = pyttsx3.init()
#         self.engine.setProperty('rate', rate)
#         self.q = queue.Queue()
#         self.last_time = 0
#         self.cooldown = 1.0

#         # Start worker thread
#         t = threading.Thread(target=self._worker, daemon=True)
#         t.start()

#     def say(self, text, force=False):
#         now = time.time()
#         if not force and (now - self.last_time) < self.cooldown:
#             return
#         self.last_time = now
#         self.q.put(text)

#     def _worker(self):
#         while True:
#             text = self.q.get()
#             if text is None:
#                 break
#             self.engine.say(text)
#             self.engine.runAndWait()
#             self.q.task_done()

#     def stop(self):
#         self.q.put(None)

# # ---------------- Main ----------------
# def main(args):
#     model = YOLO(MODEL)
#     cap = cv2.VideoCapture(PHONE_URL)  # phone feed
#     announcer = Announcer()
#     last_announced = None
#     last_detail = None
#     detected_objects = []  # <-- store all object names
#     frame_count = 0

#     if not cap.isOpened():
#         print("Cannot open phone camera stream. Check IP:PORT and network.")
#         return

#     print("Starting. Press 'c' = list all objects, 'q' = quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame. Check phone stream.")
#             break
#         h, w = frame.shape[:2]

#         frame_count += 1

#         # Run detection only every N frames
#         if frame_count % DETECT_EVERY_N_FRAMES == 0:
#             results = model(frame, imgsz=640, conf=MIN_CONF, verbose=False)
#             objs = []
#             detected_objects = []  # reset list
#             for r in results:
#                 for box in r.boxes:
#                     cls_id = int(box.cls[0])
#                     conf = float(box.conf[0])
#                     x1, y1, x2, y2 = box.xyxy[0].tolist()
#                     label = model.names[cls_id]
#                     objs.append({'label': label, 'conf': conf, 'bbox': (x1,y1,x2,y2)})
#                     detected_objects.append(label)  # collect object name

#             if objs:
#                 objs_sorted = sorted(objs, key=lambda o: -(o['bbox'][3]-o['bbox'][1]))
#                 target = objs_sorted[0]
#                 rel = bbox_center_x(target['bbox'], w)
#                 clock = rel_to_clock(rel)
#                 dist = estimate_distance_from_bbox(target['bbox'], h)
#                 msg = f"{target['label']}. {clock}. {dist:.1f} meters."
#                 detail = f"{target['label']} with {target['conf']*100:.1f}% confidence."
#                 if msg != last_announced:
#                     announcer.say(msg)
#                     last_announced, last_detail = msg, detail

#         # Show debug window
#         cv2.imshow("Phone Camera Feed", frame)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('q'):
#             break
#         elif key == ord('c'):  # announce all detected objects
#             if detected_objects:
#                 unique_objs = list(dict.fromkeys(detected_objects))  # remove duplicates, keep order
#                 announcer.say("Objects: " + ", ".join(unique_objs), force=True)
#             else:
#                 announcer.say("No objects detected.", force=True)

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default=MODEL)
#     args = parser.parse_args()
#     main(args)



import time
import threading
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import queue
from collections import OrderedDict  # new

# ---------------- Config ----------------
MODEL = "yolov8n.pt"
PHONE_URL = "http://192.168.1.70:4747/video"
TTS_RATE = 150
MIN_CONF = 0.4
SPEECH_COOLDOWN = 1.0
DETECT_EVERY_N_FRAMES = 5   # run YOLO only every 5 frames
RECENT_OBJECT_MAX_AGE = 5.0  # seconds: how long a detected object stays "recent"

# ---------------- Utilities ----------------
def bbox_center_x(bbox, frame_w):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    rel = (cx - frame_w/2) / (frame_w/2)
    return rel

def rel_to_clock(rel):
    if rel < -0.4: return "ten o'clock"
    if rel < -0.15: return "nine o'clock"
    if rel < 0.15: return "twelve o'clock"
    if rel < 0.4: return "three o'clock"
    return "two o'clock"

def estimate_distance_from_bbox(bbox, frame_h):
    x1, y1, x2, y2 = bbox
    h = abs(y2-y1)
    frac = h / frame_h
    if frac < 0.05: return 4.0
    if frac < 0.12: return 2.0
    if frac < 0.25: return 1.0
    return 0.4

# ---------------- Speech Engine ----------------
class Announcer:
    def __init__(self, rate=150):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.q = queue.Queue()
        self.last_time = 0
        self.cooldown = 1.0

        # Start worker thread
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def say(self, text, force=False):
        now = time.time()
        if not force and (now - self.last_time) < self.cooldown:
            return
        self.last_time = now
        self.q.put(text)

    def _worker(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()
            self.q.task_done()

    def stop(self):
        self.q.put(None)

# ---------------- Main ----------------
def main(args):
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(PHONE_URL)  # phone feed
    announcer = Announcer()
    last_announced = None
    last_detail = None
    # replace old list with dict storing last-seen timestamp for each label
    detected_last_seen = {}  # label -> timestamp
    frame_count = 0

    if not cap.isOpened():
        print("Cannot open phone camera stream. Check IP:PORT and network.")
        return

    print("Starting. Press 'c' (or 'C') = list all recently detected objects, 'q' = quit.")
    print("Make sure the OpenCV window has focus when pressing keys (click the window).")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check phone stream.")
            break
        h, w = frame.shape[:2]

        frame_count += 1

        # Run detection only every N frames
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            results = model(frame, imgsz=640, conf=MIN_CONF, verbose=False)
            objs = []
            # NOTE: we no longer reset a single list that disappears — instead update timestamps
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = model.names[cls_id]
                    objs.append({'label': label, 'conf': conf, 'bbox': (x1,y1,x2,y2)})
                    # record last seen timestamp for this label
                    detected_last_seen[label] = time.time()

            if objs:
                objs_sorted = sorted(objs, key=lambda o: -(o['bbox'][3]-o['bbox'][1]))
                target = objs_sorted[0]
                rel = bbox_center_x(target['bbox'], w)
                clock = rel_to_clock(rel)
                dist = estimate_distance_from_bbox(target['bbox'], h)
                msg = f"{target['label']}. {clock}. {dist:.1f} meters."
                detail = f"{target['label']} with {target['conf']*100:.1f}% confidence."
                if msg != last_announced:
                    announcer.say(msg)
                    last_announced, last_detail = msg, detail

        # Show debug window
        cv2.imshow("Phone Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in (ord('c'), ord('C')):  # announce all recently detected objects
            now = time.time()
            # collect labels seen within RECENT_OBJECT_MAX_AGE seconds, sort by most recent
            recent = sorted(
                ((label, ts) for label, ts in detected_last_seen.items() if now - ts <= RECENT_OBJECT_MAX_AGE),
                key=lambda x: -x[1]
            )
            if recent:
                recent_labels = [label for label, _ in recent]
                print("Announcing objects:", recent_labels)  # debug print
                announcer.say("Objects: " + ", ".join(recent_labels), force=True)
            else:
                print("No recent detections to announce.")
                announcer.say("No objects detected recently.", force=True)

    cap.release()
    cv2.destroyAllWindows()
    announcer.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    main(args)

for comparison:
import cv2+