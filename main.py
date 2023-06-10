import torch
import os
import cv2
import shutil
from helpers.cavetVerify import isImageContainCavetInRightDirection


class OcrOfficial:
    def __init__(self, **kwargs):
        '''
        kwargs:
            wc_path: path to model cavet detector
            wcf_path: path to model cavet fields detector
        '''
        print("Init OcrOfficial", kwargs)
        # Load model cavet detector
        self.model_cavet_detector = torch.hub.load('ultralytics/yolov5', 'custom',
                                                   path=kwargs["wc_path"], trust_repo=True)
        # Load model cavet field detector
        self.model_cavet_fields_detector = torch.hub.load('ultralytics/yolov5', 'custom',
                                                          path=kwargs["wcf_path"], trust_repo=True)
        # Load model text recognizer
        self.model_text_recognizer = None

    def set_image_config(self, **kwargs):
        '''
        kwargs:
            im_root_path: path to image need to be recognized
            save_cavet_detector_path: path to save image after cavet detector
            save_cavet_fields_detector_path: path to save image after cavet fields detector
        '''
        self.im_root_path = kwargs["im_root_path"] if "im_root_path" in kwargs else None
        self.save_cavet_detector_path = kwargs["save_cavet_detector_path"] if "save_cavet_detector_path" in kwargs else None
        self.save_cavet_fields_detector_path = kwargs[
            "save_cavet_fields_detector_path"] if "save_cavet_fields_detector_path" in kwargs else None

    def run(self):
        try:
            # Get image need to be recognized
            # Folder contain image
            im_root_path = self.im_root_path
            # List image
            im_list = os.listdir(im_root_path)
            # Remove temp folder
            shutil.rmtree(self.save_cavet_detector_path)
            # Loop image
            for im_name in im_list:
                ##################################### Process the folder terms#####################################
                # Name without extension
                im_name_without_ext = im_name.split(".")[0]
                # Read image
                im = cv2.imread(os.path.join(im_root_path, im_name))
                # Get image path
                im_path = os.path.join(im_root_path, im_name)
                # Create folder for each image to save image after cavet detector with same name
                save_cavet_detector_path = os.path.join(
                    self.save_cavet_detector_path, im_name_without_ext)
                os.makedirs(save_cavet_detector_path, exist_ok=True)
                save_cavet_detector_path = os.path.join(
                    save_cavet_detector_path, "cavet_detector")
                os.mkdir(save_cavet_detector_path)
                # Create folder for each image to save image after cavet fields detector with same name
                save_cavet_fields_detector_path = os.path.join(
                    self.save_cavet_fields_detector_path, im_name_without_ext)
                os.makedirs(save_cavet_fields_detector_path, exist_ok=True)
                save_cavet_fields_detector_path = os.path.join(
                    save_cavet_fields_detector_path, "cavet_fields_detector")
                os.mkdir(save_cavet_fields_detector_path)
                ##################################### Detection terms#####################################
                # Cavet detector
                result_cavet_detector = self.model_cavet_detector(
                    im_path, size=640)
                # Check if result is empty
                if len(result_cavet_detector.pandas().xyxy[0]) == 0:
                    print(f"Can not detect cavet in {im_name}")
                    # Remove folder
                    shutil.rmtree(os.path.join(
                        self.save_cavet_detector_path, im_name_without_ext))
                    continue
                # Coords
                cavet_detect_coords = result_cavet_detector.pandas().xyxy[0][["xmin", "ymin",
                                                                              "xmax", "ymax"]]
                # Convert tensor to numpy
                cavet_detect_coords = cavet_detect_coords.values.tolist()
                # Convert to int
                cavet_detect_coords = [list(map(int, box))
                                       for box in cavet_detect_coords]
                # Crop image
                cavet_detect_crop_im = im[cavet_detect_coords[0][1]:cavet_detect_coords[0][3],
                                          cavet_detect_coords[0][0]:cavet_detect_coords[0][2]]
                # Save image
                cv2.imwrite(os.path.join(save_cavet_detector_path,
                                         im_name), cavet_detect_crop_im)
                # If it is cavet
                if isImageContainCavetInRightDirection(im=cavet_detect_crop_im):
                    # Cavet fields detector
                    result_cavet_fields_detector = self.model_cavet_fields_detector(
                        cavet_detect_crop_im, size=640)
                    # Check if result is empty
                    if len(result_cavet_fields_detector.pandas().xyxy[0]) == 0:
                        continue
                    # Get all Classes, if there is a class that has 2 or more bounding boxes, choose the one with the highest confidence
                    classes = result_cavet_fields_detector.pandas(
                    ).xyxy[0]["name"]
                    # Get all coords
                    coords = result_cavet_fields_detector.pandas().xyxy[0][["xmin", "ymin",
                                                                            "xmax", "ymax"]]
                    # List of set (class, coords)
                    list_of_set = []
                    # Loop through classes and coords
                    for i in range(len(classes)):
                        # Get class
                        class_i = classes[i]
                        # If class is in list of set pass
                        if class_i in [x[0] for x in list_of_set]:
                            continue
                        # Get coords
                        coords_i = coords.iloc[i].to_numpy()
                        # Append to list of set
                        list_of_set.append((class_i, coords_i))
                    # Loop through list of set
                    for i in range(len(list_of_set)):
                        # Get class
                        class_i = list_of_set[i][0]
                        # Get coords
                        coords_i = list_of_set[i][1]
                        # Convert to int
                        coords_i = list(map(int, coords_i))
                        # Crop image
                        cavet_fields_detect_crop_im = cavet_detect_crop_im[coords_i[1]:coords_i[3],
                                                                           coords_i[0]:coords_i[2]]
                        # Save image
                        cv2.imwrite(os.path.join(save_cavet_fields_detector_path,
                                                 class_i + "_" + im_name), cavet_fields_detect_crop_im)
                else:
                    continue
            # Cavet detector
            pass
        except Exception as e:
            print(e)


if __name__ == "__main__":
    ocr = OcrOfficial(
        wc_path="./weights/CavetDetector_v1.pt",
        wcf_path="./weights/CavetFieldsDetecotor_v1.pt"
    )
    ocr.set_image_config(
        im_root_path="./run/detect/images",
        save_cavet_detector_path="./run/temp/",
        save_cavet_fields_detector_path="./run/temp/"
    )
    ocr.run()
