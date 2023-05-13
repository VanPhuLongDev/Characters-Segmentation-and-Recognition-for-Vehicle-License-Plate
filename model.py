import argparse
from pathlib import Path
import tensorflow as tf
import torch
import numpy as np
import cv2
from imutils import perspective
import numpy as np
from skimage.filters import threshold_local
import imutils
from keras.models import load_model

best_path = "src/util/best.pt"
model_detect_frame = torch.hub.load('ultralytics/yolov5', 'custom',
                        path=best_path, force_reload=True)
model_detect_text = load_model("src/util/trained_model_6.h5", compile=False)
def rotate_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    x,y,w,h = cv2.boundingRect(max_cnt)
    
    rect = cv2.minAreaRect(max_cnt)
    ((cx,cy),(cw,ch),angle) = rect
    
    M = cv2.getRotationMatrix2D((cx,cy), angle-90, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    x,y,w,h = cv2.boundingRect(max_cnt)
    cropped = rotated[y:y+h, x:x+w]
    return cropped


def getPlateTextFromImage(imgPath):
    image = cv2.imread(imgPath)
    results = model_detect_frame(image)
    df = results.pandas().xyxy[0]
    for obj in df.iloc:
        xmin = float(obj['xmin'])
        xmax = float(obj['xmax'])
        ymin = float(obj['ymin'])
        ymax = float(obj['ymax'])
    coord = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    LpRegion = perspective.four_point_transform(image, coord)
    
    LpRegion = rotate_and_crop(LpRegion)
    image = LpRegion.copy()
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    # adaptive threshold
    T = threshold_local(V, 35, offset=5, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, width=600)

    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")
    total_pixels = thresh.shape[0] * thresh.shape[1]
    lower = total_pixels // 90
    upper = total_pixels // 20
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    cnts, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    boundingBoxes = np.array(boundingBoxes)
    mean_w = np.mean(boundingBoxes[:, 2])
    mean_h = np.mean(boundingBoxes[:, 3])
    mean_y = np.mean(boundingBoxes[:,1])
    threshold_w = mean_w * 1.5
    threshold_h = mean_h * 1.5
    new_boundingBoxes = boundingBoxes[(boundingBoxes[:, 2] < threshold_w) & (boundingBoxes[:, 3] < threshold_h)]
    line1 = []
    line2 = []
    for box in new_boundingBoxes:
        x,y,w,h  =box
        if y > mean_y * 1.2:
            line2.append(box)
        else:
            line1.append(box)

    line1 = sorted(line1, key=lambda box: box[0])
    line2 = sorted(line2, key=lambda box: box[0])
    boundingBoxes = line1+line2

    img_with_boxes = imutils.resize(image.copy(), width=600)
    image = imutils.resize(image.copy(), width=600)
    for bbox in boundingBoxes:
        x, y, w, h = bbox
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Character Recognition

    chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'
    ]
    vehicle_plate = ""
    characters = []

    
    for rect in boundingBoxes:
        x, y, w, h = rect

        character = mask[y:y+h, x:x+w]
        character = cv2.bitwise_not(character)
        rows = character.shape[0]
        columns = character.shape[1]
        paddingY = (128 - rows) // 2 if rows < 128 else int(0.17 * rows)
        paddingX = (
            128 - columns) // 2 if columns < 128 else int(0.45 * columns)
        character = cv2.copyMakeBorder(character, paddingY, paddingY,
                                    paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        character = cv2.cvtColor(character, cv2.COLOR_GRAY2RGB)
        character = cv2.resize(character, (128, 128))
        character = character.astype("float") / 255.0
        characters.append(character)
    characters = np.array(characters)
    probs = model_detect_text.predict(characters)
    for prob in probs:
        idx = np.argsort(prob)[-1]
        vehicle_plate += chars[idx]
    return vehicle_plate

def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./samples/1.jpg')

    return arg.parse_args()

args = get_arguments()
img_path = Path(args.image_path)
getPlateTextFromImage(img_path)