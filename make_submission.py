import cv2
import pandas as pd
import torch
from PytorchWildlife.models import detection as pw_detection
from nn_models.classifier import processor, swin_model


def xyxy2xcycwh(coordinates, shape):
    h, w = shape
    xc = (coordinates[2] + coordinates[0]) / (2 * w)
    yc = (coordinates[3] + coordinates[1]) / (2 * h)

    w_bbox = (coordinates[2] - coordinates[0]) / w
    h_bbox = (coordinates[2] - coordinates[0]) / h
    return ','.join(list(
        map(
            str,
            [xc, yc, w_bbox, h_bbox]
        )
    ))

WRONG_DETECTION = 2

path = r'C:\DATA\images'
device = 'cuda:0'
detection_model = pw_detection.MegaDetectorV5(device='cuda:0', pretrained=True)

res = detection_model.batch_image_detection(path, batch_size=4, conf_thres=0.2)

all_data = []

df = pd.DataFrame(data=all_data, columns=['Name', 'Bbox', 'Class'])

for detection in res:
    xyxy = detection['detections'].xyxy
    name = detection['img_id']
    im = cv2.imread(name)
    shape = im.shape[:2]
    batch = []
    current_photo_data = []
    for bbox in xyxy:
        x1, y1, x2, y2 = list(map(int, bbox))
        cropped_image = im[y1:y2, x1:x2]
        input = processor(images=cropped_image, return_tensors="pt")

        batch.append(input['pixel_values'])

    input = torch.cat(batch).to(device)

    input = {'pixel_values': input}
    with torch.no_grad():
        outputs = swin_model(**input)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1)
    for bbox, label in zip(xyxy, predicted_class_idx):
        label = label.item()
        if label == WRONG_DETECTION:
            continue
        else:
            xcycwh = xyxy2xcycwh(bbox, shape)
            data_entry = {'Name': name[len(path) + 1:],
                         'Bbox': xcycwh,
                         'Class': label}
            current_photo_data.append(data_entry)

    df1 = pd.DataFrame(data=current_photo_data, columns=data_entry.keys())
    df = pd.concat([df, df1], ignore_index=True)


