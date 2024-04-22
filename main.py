from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import numpy as np
import pandas as pd


import os

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    results = {}
    # vehicle_track_ids = [] 
    frame_nmr = -1
    vehicles = [2, 3, 5, 7]

    # object tracker
    mot_tracker = Sort()

    # load models
    print("Loading COCO model...")
    coco_model = YOLO('yolov8n.pt')
    print("COCO model loaded.")
    
    print("Loading license plate detector model...")
    license_plate_detector = YOLO(r"license_plate_detector.pt")
    print("License plate detector model loaded.")

    # load video
    cap = cv2.VideoCapture(r"sample.mp4")

    
    ret = True
    while ret and frame_nmr < 50:
        frame_nmr += 1
        print("-----------------------------------")
        print(f"| Processing frame {frame_nmr}... |")
        print("-----------------------------------") 
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # track vehicles
            track_ids = mot_tracker.update(np.array(detections_))

            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                        
                        # Debug check: Print the license plate text for each frame
                        print(" -------------------------------------------------------------- ")
                        print(f"Frame {frame_nmr}: License plate text - {license_plate_text}")
                        print(" -------------------------------------------------------------- ")

    # write results
    print("Writing results to CSV...")
    write_csv(results, './test.csv')
    print("Results written to CSV.")



data = pd.read_csv(r'test.csv')
sorted_data = data.sort_values(by=['car_id', 'license_number_score'], ascending=[True, False])
unique_license_numbers = sorted_data.drop_duplicates(subset='car_id', keep='first')
result = unique_license_numbers[['car_id', 'license_number']]
print(result)


write_csv(results, './test_max_conf.csv')
