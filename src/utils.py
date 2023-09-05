import os

import numpy as np
import supervision as sv


def create_dir(path: str) -> None:
    if not os.path.exists(path):
       os.makedirs(path)
    
    
def get_name(path: str) -> str:
    name = os.path.basename(str(path)).split(".")
    return ".".join(name[:-1])


def save_yolo_detection(path, name, detections, width, height) -> None:
    # save detections to file
    with open(path + 'obj_train_data/' + name + ".txt", "w") as f:

        for xyxy, class_id in zip(detections.xyxy, detections.class_id):

            yolo_label = label_to_yolo(xyxy, width, height)
            yolo_label = [str(x) for x in yolo_label]
            f.write(str(class_id) + " " + " ".join(yolo_label) + "\n")

        if len(detections.xyxy) == 0:
            f.write("\n")
          
        
def label_to_yolo(label, image_w, image_h):

    x_min, x_max = label[0], label[2]
    y_min, y_max = label[1], label[3]

    label_w = x_max - x_min
    label_h = y_max - y_min
    
    center_x = ( x_min + label_w/2 ) / image_w
    center_y = ( y_min + label_h/2 ) / image_h
    
    return center_x, center_y, label_w / image_w, label_h / image_h


def combine_detections(detections_list, overwrite_class_ids):
    if len(detections_list) == 0:
        return sv.Detections.empty()

    if overwrite_class_ids is not None and len(overwrite_class_ids) != len(
        detections_list
    ):
        raise ValueError(
            "Length of overwrite_class_ids must match the length of detections_list."
        )

    xyxy = []
    mask = []
    confidence = []
    class_id = []
    tracker_id = []

    for idx, detection in enumerate(detections_list):
        xyxy.append(detection.xyxy)

        if detection.mask is not None:
            mask.append(detection.mask)

        if detection.confidence is not None:
            confidence.append(detection.confidence)

        if detection.class_id is not None:
            if overwrite_class_ids is not None:
                # Overwrite the class IDs for the current Detections object
                class_id.append(
                    np.full_like(
                        detection.class_id, overwrite_class_ids[idx], dtype=np.int64
                    )
                )
            else:
                class_id.append(detection.class_id)

        if detection.tracker_id is not None:
            tracker_id.append(detection.tracker_id)

    xyxy = np.vstack(xyxy)
    mask = np.vstack(mask) if mask else None
    confidence = np.hstack(confidence) if confidence else None
    class_id = np.hstack(class_id) if class_id else None
    tracker_id = np.hstack(tracker_id) if tracker_id else None

    return sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )
