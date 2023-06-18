import sys, warnings
sys.path.insert(0, "yolov5_face")
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import os
import cv2
import shutil
import torch

from torchvision import transforms
from model_arch.backbones import get_model
from model_arch.backbones.iresnet import iresnet100
from yolov5_face.models.experimental import attempt_load
from yolov5_face.utils.datasets import letterbox
from yolov5_face.utils.general import check_img_size, non_max_suppression_face, scale_coords

class FaceRecognitionTrainer:
    def __init__(self, full_train_dir, add_train_dir, faces_save_dir, feat_save_dir, is_add_user):
        """
        Initialize the FaceRecognitionTrainer class.

        Args:
            full_train_dir (str): Directory containing full training data.
            add_train_dir (str): Directory containing additional training data.
            faces_save_dir (str): Directory to save face datasets.
            feat_save_dir (str): Directory to save face embeddings.
            is_add_user (bool): Mode: add user or full training.
        """
        
        self.full_train_dir = full_train_dir
        self.add_train_dir = add_train_dir
        self.faces_save_dir = faces_save_dir
        self.feat_save_dir = feat_save_dir
        self.is_add_user = is_add_user
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detection_model = attempt_load("model_arch/weights/yolov5m-face.pt")
        self.face_embedding_model = iresnet100()
        self.face_embedding_model.load_state_dict(torch.load("model_arch/backbones/backbone.pth", map_location=self.device))
        self.face_embedding_model.to(self.device)
        self.face_embedding_model.eval()
        self.face_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def resize_image(self, img0, img_size):
        """
        Resize the image while maintaining aspect ratio and applying letterboxing.

        Args:
            img0 (numpy.ndarray): Input image as a NumPy array.
            img_size (int): Target image size.

        Returns:
            torch.Tensor: Resized and preprocessed image as a PyTorch tensor.
        """
        h0, w0 = img0.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=self.face_detection_model.stride.max())  # check img_size
        img = letterbox(img0, new_shape=imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img

    def detect_faces(self, input_image):
        """
        Detect faces in the input image using the YOLOv5 face detection model.

        Args:
            input_image (numpy.ndarray): Input image as a NumPy array.

        Returns:
            numpy.ndarray: Detected bounding boxes of faces.
        """
        size_convert = 256
        conf_thres = 0.4
        iou_thres = 0.5

        img = self.resize_image(input_image.copy(), size_convert)

        with torch.no_grad():
            pred = self.face_detection_model(img[None, :])[0]

        det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
        bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())

        return bboxs

    def extract_face_embedding(self, face_image, training=True):
        """
        Extract the face embedding (feature vector) from the face image using the ResNet-100 backbone model.

        Args:
            face_image (numpy.ndarray): Face image as a NumPy array.
            training (bool): Flag indicating whether the model is used for training or inference.

        Returns:
            numpy.ndarray: Face embedding (feature vector).
        """
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = self.face_preprocess(face_image).to(self.device)

        with torch.no_grad():
            if training:
                emb_img_face = self.face_embedding_model(face_image[None, :])[0].cpu().numpy()
            else:
                emb_img_face = self.face_embedding_model(face_image[None, :]).cpu().numpy()

        images_emb = emb_img_face / np.linalg.norm(emb_img_face)
        return images_emb

    def read_features(self, root_feature_path):
        """
        Read the saved face embeddings and associated image names from a file.

        Args:
            root_feature_path (str): Path to the root feature file.

        Returns:
            tuple: Tuple containing the loaded image names and embeddings, or None if the file is not found.
        """
        try:
            data = np.load(root_feature_path + ".npz", allow_pickle=True)
            images_name = data["arr1"]
            images_emb = data["arr2"]

            return images_name, images_emb
        except:
            return None

    def training(self):
        """
        Perform face recognition training by detecting faces in the images, extracting face embeddings,
        and saving the embeddings to a file.
        """
        images_name = []
        images_emb = []

        if self.is_add_user:
            source = self.add_train_dir
        else:
            source = self.full_train_dir

        for name_person in os.listdir(source):
            person_image_path = os.path.join(source, name_person)

            # split folder name into last name and first name
            last_name, first_name = name_person.split("_")

            person_face_path = os.path.join(self.faces_save_dir, name_person)
            os.makedirs(person_face_path, exist_ok=True)

            for image_name in os.listdir(person_image_path):
                if image_name.endswith(("png", 'jpg', 'jpeg')):
                    image_path = os.path.join(person_image_path, image_name)
                    input_image = cv2.imread(image_path)

                    bboxs = self.detect_faces(input_image)

                    for i in range(len(bboxs)):
                        number_files = len(os.listdir(person_face_path))
                        x1, y1, x2, y2 = bboxs[i]
                        face_image = input_image[y1:y2, x1:x2]
                        path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")

                        cv2.imwrite(path_save_face, face_image)
                        images_emb.append(self.extract_face_embedding(face_image, training=True))
                        images_name.append(f"{first_name.capitalize()} {last_name.capitalize()}")

        images_emb = np.array(images_emb)
        images_name = np.array(images_name)

        features = self.read_features(self.feat_save_dir)
        if features is not None and not self.is_add_user:
            old_images_name, old_images_emb = features
            images_name = np.hstack((old_images_name, images_name))
            images_emb = np.vstack((old_images_emb, images_emb))
            print("Embeddings done.")

        np.savez_compressed(self.feat_save_dir, arr1=images_name, arr2=images_emb)

        if self.is_add_user:
            for sub_dir in os.listdir(self.add_train_dir):
                dir_to_move = os.path.join(self.add_train_dir, sub_dir)
                shutil.move(dir_to_move, self.full_train_dir, copy_function=shutil.copytree)
        
        print("Updated features successfully")

    @staticmethod
    def parse_opt():
        """
        Parse command line arguments.

        Returns:
            argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--full-train-dir', type=str, default='./dataset/trained-data/',
                            help='Directory containing full training data')
        parser.add_argument('--add-train-dir', type=str, default='./dataset/add-train-data/',
                            help='Directory containing additional training data')
        parser.add_argument('--faces-save-dir', type=str, default='./dataset/face-datasets/',
                            help='Directory to save face datasets')
        parser.add_argument('--feat-save-dir', type=str, default='./static/features',
                            help='Directory to save face embeddings')
        parser.add_argument('--is-add-user', type=bool, default=True, help='Mode: add user or full training')

        opt = parser.parse_args()
        return opt

    @staticmethod
    def main(opt):
        """
        Main function to instantiate the FaceRecognitionTrainer class and start the training process.

        Args:
            opt (argparse.Namespace): Parsed command line arguments.
        """
        trainer = FaceRecognitionTrainer(
            opt.full_train_dir,
            opt.add_train_dir,
            opt.faces_save_dir,
            opt.feat_save_dir,
            opt.is_add_user
        )
        trainer.training()

if __name__ == "__main__":
    opt = FaceRecognitionTrainer.parse_opt()
    FaceRecognitionTrainer.main(opt)
