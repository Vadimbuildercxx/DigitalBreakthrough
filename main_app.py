import json
import os
import tempfile

import PIL.Image
import cv2
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
import shutil
import supervision as sv
import gradio as gr
from zipfile import ZipFile


import numpy as np
from io import BytesIO

from nn_models.classifier import swin_model, processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dot_annotator = sv.DotAnnotator(radius=6)
box_annotator = sv.BoxAnnotator(thickness=4)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)

os.makedirs(os.path.join("..", "temp"), exist_ok=True)

detection_model = None
classification_model = None

def files_to_batch(file):

    all_files = []


    zip_file = ZipFile(file)
    for file_name in sorted(zip_file.namelist()):
        file_contents = zip_file.read(file_name)
        all_files.append(file_contents)

    batch = []

    for image_bytes in all_files:
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        batch.append(image)

    return batch

def create_byte_generator(results):
    for fname, pil_img in results:
        with BytesIO() as output:
            pil_img = PIL.Image.fromarray(pil_img[..., ::-1])
            pil_img.save(output, format="PNG")
            yield fname, output.getvalue()

def load_models(det, clf, wpath=None, wclass=None):
    global detection_model, classification_model
    if det != "None":
        if det == "MegaDetectorV6":
            detection_model = pw_detection.__dict__[det](device=DEVICE, pretrained=True)
        else:
            detection_model = pw_detection.__dict__[det](device=DEVICE, pretrained=True)
    if clf != 'None':
        if clf == 'SWIN':
            classification_model = swin_model
    return "Детектор: {}\nКлассификатор: {}".format(det, clf)


def single_image_detection(input_img, det_conf_thres, img_index=None):
    """Performs detection on a single image and returns an annotated image.

    Args:
        input_img (PIL.Image): Input image in PIL.Image format defaulted by Gradio.
        det_conf_thre (float): Confidence threshold for detection.
        clf_conf_thre (float): Confidence threshold for classification.
        img_index: Image index identifier.
    Returns:
        annotated_img (PIL.Image.Image): Annotated image with bounding box instances.
    """

    input_img = np.array(input_img)

    results_det = detection_model.single_image_detection(input_img,
                                                         img_path=img_index,
                                                         conf_thres=det_conf_thres)
    if not results_det:
        return input_img

    batch = []

    for bbox in results_det['detections'].xyxy:
        x1, y1, x2, y2 = list(map(int, bbox))
        cropped_image = input_img[y1:y2, x1:x2]
        input = processor(images=cropped_image, return_tensors="pt")
        batch.append(input['pixel_values'])

    input = torch.cat(batch).to(DEVICE)

    with torch.no_grad():
        outputs = swin_model(**{'pixel_values': input})
    logits = outputs.logits
    labels = logits.argmax(-1).tolist()
    for bbox, label in zip(results_det['detections'].xyxy, labels):
        x1, y1, x2, y2 = list(map(int, bbox))
        if label == 0:
            color = (0, 0, 255)
        elif label == 1:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.rectangle(input_img, (x1, y1), (x2, y2), color, 5)

    return PIL.Image.fromarray(input_img)


def batch_detection(zip_file, timelapse, det_conf_thres):
    """Perform detection on a batch of images from a zip file and return path to results JSON.

    Args:
        zip_file (File): Zip file containing images.
        det_conf_thre (float): Confidence threshold for detection.
        timelapse (boolean): Flag to output JSON for timelapse.
        clf_conf_thre (float): Confidence threshold for classification.

    Returns:
        json_save_path (str): Path to the JSON file containing detection results.
    """
    # Clean the temp folder if it contains files

    extract_path = os.path.join("..", "temp", "zip_upload")

    os.makedirs('./runs', exist_ok=True)

    zip_data = zip_file.name[:-4] + '_PROCESSED'
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)

    json_save_path = os.path.join(extract_path, "results.json")

    images = files_to_batch(zip_file)

    with ZipFile(zip_file.name) as zfile:
        zfile.extractall(extract_path)

        # Check the contents of the extracted folder
        extracted_files = os.listdir(extract_path)

    if len(extracted_files) == 1 and os.path.isdir(os.path.join(extract_path, extracted_files[0])):
        tgt_folder_path = os.path.join(extract_path, extracted_files[0])
    else:
        tgt_folder_path = extract_path
    try:
        os.makedirs('./temp_zip/no_detections')
        os.makedirs('./temp_zip/has_detections/good_photos')
        os.makedirs('./temp_zip/has_detections/bad_photos')

        det_results = detection_model.batch_image_detection(tgt_folder_path, batch_size=16, conf_thres=det_conf_thres,
                                                            id_strip=tgt_folder_path)

        images_with_detections = []
        dct = {}
        for i, det_res in enumerate(det_results):
            images_with_detections.append(det_res['img_id'])
            im = images[i]
            batch = []
            for bbox in det_res['detections'].xyxy:
                x1, y1, x2, y2 = list(map(int, bbox))
                cropped_image = im[y1:y2, x1:x2]
                input = processor(images=cropped_image, return_tensors="pt")
                batch.append(input['pixel_values'])

            input = torch.cat(batch).to(DEVICE)

            with torch.no_grad():
                outputs = swin_model(**{'pixel_values': input})
            logits = outputs.logits
            labels = logits.argmax(-1).tolist()
            print(labels)
            dct[det_res['img_id']] = {'bboxes': det_res['detections'].xyxy.tolist(),
                                      'labels': ['Bad' if label == 0 else 'Good' if label == 1 else 'Detection_mistake' for label in labels]}

            for bbox, label in zip(det_res['detections'].xyxy, labels):

                x1, y1, x2, y2 = list(map(int, bbox))
                if label == 0:
                    color = (0, 0, 255)
                elif label == 1:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 4)

            if any(labels):
                cv2.imwrite(f'./temp_zip/has_detections/good_photos/{extracted_files[i]}', im)
            else:
                cv2.imwrite(f'./temp_zip/has_detections/bad_photos/{extracted_files[i]}', im)

        no_detection_images = list(set(extracted_files) - set(images_with_detections))
        for image in no_detection_images:
            cv2.imwrite(f'./temp_zip/no_detections/{image}', images[extracted_files.index(image)])
            dct[image] = {'bboxes': None, 'labels': None}
        with open('./temp_zip/info.json', 'w') as js:
            json.dump(dct, js)

        shutil.make_archive(zip_data, 'zip', './temp_zip')
    except:
        pass
    finally:
        shutil.rmtree('./temp_zip')

    return zip_data + '.zip'


with gr.Blocks() as demo:
    gr.Markdown("# Автоматическая фильтрация фотографий животных")
    with gr.Row():
        det_drop = gr.Dropdown(
            ["MegaDetectorV5", "MegaDetectorV6",],
            label="Модель детекции животных",
            value="MegaDetectorV5"  # Default
        )
        clf_drop = gr.Dropdown(
            ["SWIN"],
            interactive=True,
            label="Модель классификации",
            value="SWIN"
        )
    with gr.Column():
        load_but = gr.Button("Загрузить выбранные модели")
        load_out = gr.Text("Выберите модели", label="Загруженные модели:")

    with gr.Tab("Обработать одну фотографию"):
        with gr.Row():
            with gr.Column():
                sgl_in = gr.Image(type="pil")
                sgl_conf_sl_det = gr.Slider(0, 1, label="Порог уверенности детекции", value=0.2)
            sgl_out = gr.Image()
        sgl_but = gr.Button("Начать обработку")

    with gr.Tab("Обработка ZIP-архива"):
        with gr.Row():
            with gr.Column():
                bth_in = gr.File(label="Загрузите zip")
                # The timelapse checkbox is only visible when the detection model is not HerdNet
                chck_timelapse = gr.Checkbox(label="Generate timelapse JSON", visible=False)
                bth_conf_sl = gr.Slider(0, 1, label="Порог уверенности детекции", value=0.2)
            bth_out = gr.File(label="Отсортированный архив", height=200, type='binary', file_types=['file'])
        bth_but = gr.Button("Начать обработку")

    load_but.click(load_models, inputs=[det_drop, clf_drop],
                   outputs=load_out)

    sgl_but.click(single_image_detection, inputs=[sgl_in, sgl_conf_sl_det], outputs=sgl_out)
    bth_but.click(batch_detection, inputs=[bth_in, chck_timelapse, bth_conf_sl], outputs=bth_out)


if __name__ == "__main__":
    demo.launch(share=True)