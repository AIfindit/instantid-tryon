import torch
import numpy as np
import cv2
from PIL import Image

def encode_face(image, face):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_crop = image[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (224, 224))
    face_crop = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    face_tensor = torch.tensor(np.array(face_crop)).permute(2, 0, 1).float() / 255
    face_tensor = face_tensor.unsqueeze(0)
    return face_tensor
