import sys

import cv2
from tqdm import tqdm

sys.path.append('/app/')
import src.utils as utils


def process(image_paths, classes, model, BOX_TRESHOLD, TEXT_TRESHOLD, SOURCE_DIRECTORY_PATH):
    
    # создание необходимых директорий
    utils.create_dir(SOURCE_DIRECTORY_PATH)
    utils.create_dir(SOURCE_DIRECTORY_PATH + 'obj_train_data/')

    detections = []

    for image_path in tqdm(image_paths):

        image = cv2.imread(str(image_path))

        height, width, depth = image.shape

        name = utils.get_name(image_path)

        detections_list = []

        for i, description in enumerate(classes.keys()):

            detection = model.predict_with_classes(
                image=image,
                classes=[description],
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

            detection = detection[detection.confidence > classes[description]]

            # drop potential detections with area close to area of whole image
            detection = detection[(detection.area / (height * width)) < 0.9 ]

            detections_list.append(detection)

        if detections_list:

            detections_list = utils.combine_detections(
                        detections_list, overwrite_class_ids=range(len(detections_list))
                    )

            # drop potential double detections
            detections_list = detections_list.with_nms(threshold=0.5, class_agnostic=True)

            # save detections to file
            utils.save_yolo_detection(SOURCE_DIRECTORY_PATH, name, detections_list, width, height)

            detections.append(detections_list)
        else:
            detections.append(None)
            
    return detections
