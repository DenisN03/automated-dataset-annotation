import sys

import cv2
import numpy as np
from tqdm import tqdm
import supervision as sv

sys.path.append('/app/')
import src.utils as utils


def process_gd(image_paths, classes, model, BOX_TRESHOLD, TEXT_TRESHOLD, SOURCE_DIRECTORY_PATH):
    
    # create dirs
    utils.create_dir(SOURCE_DIRECTORY_PATH)
    utils.create_dir(SOURCE_DIRECTORY_PATH + 'obj_train_data/')

    detections = {}

    for image_path in tqdm(image_paths):
        
        image_path = str(image_path)

        image = cv2.imread(image_path)

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
            sv_detections = detections_list.with_nms(threshold=0.5, class_agnostic=True)

            # save detections to file
            utils.save_yolo_detection(SOURCE_DIRECTORY_PATH, name, sv_detections, width, height)

            detections[image_path] = sv_detections
        else:
            detections[image_path] = None
            
    return detections


def process_sam(detections, model, folder, save_mask: bool, SOURCE_DIRECTORY_PATH):
    
    utils.create_dir(SOURCE_DIRECTORY_PATH)
    utils.create_dir(SOURCE_DIRECTORY_PATH + f'{folder}/')
    
    for image_path, detection in tqdm(detections.items()):
        
        image_path = str(image_path)
        
        name = utils.get_name(image_path)
        
        image = cv2.imread(image_path)
        
        height, width, depth = image.shape
    
        detection.mask = utils.segment(
                    sam_predictor=model,
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detection.xyxy
                )
        
        polygons = []
        for mask, class_id in zip(detection.mask, detection.class_id):
            polygon = sv.dataset.ultils.approximate_mask_with_polygons(
                    mask=mask,
                    min_image_area_percentage=0.00, # 0.002
                    max_image_area_percentage=1.00, # 0.80
                    approximation_percentage=0.75,
                )
            polygons.append([class_id, polygon])
        
        utils.save_yolo_polygon(SOURCE_DIRECTORY_PATH + f'{folder}/', name, polygons, width, height)
        
        if save_mask == False:
            detection.mask = []
    
    return detections
