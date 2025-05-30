import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict
import pyrealsense2 as rs


def inference(
        model,
        task,
        show_output=True,
        count=False,
        show_tracks=False,
):
    # History for tracking lines
    track_history = defaultdict(lambda: [])

    # History for unique object IDs per class (used in tracking count)
    seen_ids_per_class = defaultdict(set)

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        start_time = time.time()
        class_counts = defaultdict(int)

        # Inference
        if task == "track":
            results = model.track(frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
        elif task == "detect":
            results = model.predict(frame, conf=0.5)
        else:
            raise ValueError("Invalid task. Use 'detect' or 'track'.")

        end_time = time.time()
        annotated_frame = results[0].plot()

        if results[0].boxes and results[0].boxes.cls is not None:
            boxes = results[0].boxes.xywh.cpu()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            names = results[0].names

            if task == "track" and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                    x, y, w, h = box
                    class_name = names[cls_id]

                    # Save this ID for unique counting
                    if count:
                        seen_ids_per_class[class_name].add(track_id)

                    # Draw tracking lines
                    if show_tracks:
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            elif task == "detect" and count:
                for cls_id in class_ids:
                    class_counts[names[cls_id]] += 1

        # Draw class counts in bottom-left corner
        if count:
            x0, y0 = 10, annotated_frame.shape[0] - 80
            if task == "track":
                for i, (cls_name, ids) in enumerate(seen_ids_per_class.items()):
                    label = f"{cls_name}: {len(ids)}"
                    y = y0 + i * 25
                    cv2.putText(annotated_frame, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif task == "detect":
                for i, (cls_name, total) in enumerate(class_counts.items()):
                    label = f"{cls_name}: {total}"
                    y = y0 + i * 25
                    cv2.putText(annotated_frame, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw FPS
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if show_output:
            cv2.imshow("Raspbery Pi x YOLO11 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


# Example usage
model = YOLO("yolo11n.mnn", task="detect")
# model = YOLO("yolo11n-seg_openvino_model", task="segment")
# model = YOLO("yolo11n-pose_ncnn_model", task="pose")

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

inference(
    model,
    task="track",
    show_output=True,
    count=True,
    show_tracks=False,
)
