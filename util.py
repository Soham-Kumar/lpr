import string
# import easyocr
from paddleocr import PaddleOCR
# import cv2
import numpy as np


percentage = 0.6
centre_dist = 30

# Mapping dictionaries for character conversion
dict_char_to_int = {
    'O': '0',
    'I': '1',
    'Z': '2',
    'J': '3',
    'A': '4',
    'S': '5',
    'G': '6'
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',

    '2': 'Z',
    '3': 'J',
    '4': 'A',
    '5': 'S',
    '6': 'G'
}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


import string

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    The valid formats are:
    - 7 characters: ABC4567 or WD4567C
    - 8 characters: QAA4567C, SAB4567C, or KV4567B

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) == 7:
        # Check for format: first 3 are letters and next 4 are numbers OR first 2 are letters, followed by 4 numbers and 1 letter
        if (text[:3].isalpha() and text[3:].isdigit()) or \
           (text[:2].isalpha() and text[2:6].isdigit() and text[6].isalpha()):
            return all(c.upper() in string.ascii_uppercase for c in text if c.isalpha())
    elif len(text) == 8:
        # Check for format: first 3 are letters, next 4 are numbers, last is a letter
        if text[:3].isalpha() and text[3:7].isdigit() and text[7].isalpha():
            return all(c.upper() in string.ascii_uppercase for c in text if c.isalpha())
    return False



def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_



# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image.
#     """

#     detections = reader.readtext(license_plate_crop)

#     for detection in detections:
#         bbox, text, score = detection

#         text = text.upper().replace(' ', '')
#         return text, score

#         # if license_complies_format(text):
#         #     return format_license(text), score

#     # return None, None



# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image using PaddleOCR.
#     """
#     ocr = PaddleOCR(use_angle_cls=True, lang="en")  # use_angle_cls will help in correcting orientation

#     result = ocr.ocr(license_plate_crop, cls=True)

#     if result and result[0]:
#         print(result)
#         if len(result) == 1:
#             for line in result[0]:
#                 text, confidence = line[1]
#                 text = text.upper().replace(' ', '')
#                 return text, confidence
#         # elif len(result) > 1:

#             # if license_complies_format(text):
#             #     return format_license(text), confidence
#     else:
#         return None, None  # Return None if no text was detected


# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image using PaddleOCR.
#     """
    

#     ocr = PaddleOCR(use_angle_cls=True, lang="en")  # use_angle_cls will help in correcting orientation

#     result = ocr.ocr(license_plate_crop, cls=True)

#     if result and result[0]:
#         concatenated_text = ""
#         confidence = 0
#         for line in result[0]:
#             text, conf = line[1]
#             text = text.upper().replace(' ', '')
#             concatenated_text += text
#             confidence += conf
#         return concatenated_text, confidence / len(result[0])
#     return None, None  # Return None if no text was detected



def calculate_center(box):
    """Calculate the center of a bounding box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image using PaddleOCR.
    """
    ocr = PaddleOCR(use_angle_cls=True, lang="en")  # use_angle_cls will help in correcting orientation

    result = ocr.ocr(license_plate_crop, cls=True)

    if result and result[0]:
        center_image = (license_plate_crop.shape[1] / 2, license_plate_crop.shape[0] / 2)
        concatenated_text = ""
        confidence = 0
        for line in result[0]:
            points, (text, conf) = line
            text = text.upper().replace(' ', '')
            x_coordinates = [point[0] for point in points]
            y_coordinates = [point[1] for point in points]
            text_bbox = [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]
            center_text = calculate_center(text_bbox)
            distance = np.sqrt((center_text[0] - center_image[0])**2 + (center_text[1] - center_image[1])**2)
            print("=====================")
            print(distance)
            print("=====================")
            if distance <= centre_dist:
                concatenated_text += text
                confidence += conf
        if concatenated_text:
            return concatenated_text, confidence / len(result[0])
    return None, None  # Return None if no text was detected or all texts are filtered out




def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # lisence plate is inside the car
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
