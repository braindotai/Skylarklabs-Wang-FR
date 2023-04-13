
import os
import sys

import numpy as np
import torch
import cv2
from face_landmark.facelandmark import FaceLandmark

# create face landmark extractor and initialize it
landmark_path = os.path.join(os.path.dirname(__file__), 'face_landmark/landmark.tar')

model_landmark = FaceLandmark()
model_landmark.load_state_dict(torch.load(landmark_path, map_location='cpu')['state_dict'])
model_landmark.eval()


def get_face_landmark(gray_image, bounding_box):
    """
        Description:
            get face landmark in gray image with face rect

        Args:
            gray_image:input gray image
            bounding_box:face rect
    """

    image = gray_image
    box = bounding_box

    nHeight, nWidth = image.shape

    rLeftMargin = 0.05
    rTopMargin = 0.00
    rRightMargin = 0.05
    rBottomMargin = 0.10

    # rW = box[2] - box[0]
    # rH = box[3] - box[1]
    rW = box[2]
    rH = box[3]

    rX = box[0]
    rY = box[1]

    #get image range to get face landmark from face rect
    iExFaceX = int(rX - rLeftMargin * rW)
    iExFaceY = int(rY - rTopMargin * rH)
    iExFaceW = int((1 + (rLeftMargin + rRightMargin)) * rW)
    iExFaceH = int((1 + (rTopMargin + rBottomMargin)) * rH)

    iExFaceX = np.clip(iExFaceX, 0, nWidth - 1)
    iExFaceY = np.clip(iExFaceY, 0, nHeight - 1)
    iExFaceW = np.clip(iExFaceX + iExFaceW, 0, nWidth - 1) - iExFaceX
    iExFaceH = np.clip(iExFaceY + iExFaceH, 0, nHeight - 1) - iExFaceY

    #crop face image in range to face landmark
    image = image[iExFaceY:iExFaceY+iExFaceH, iExFaceX:iExFaceX+iExFaceW]
    #normalize croped face image
    image = cv2.resize(image, (64, 64), cv2.INTER_LINEAR)
    image = image / 256
    image = torch.from_numpy(image.astype(np.float32))
    #convert mask_align_image from type [n,n] to [1,1,n,n]
    image = image.unsqueeze(0).unsqueeze(0)

    #get landmark fron croped face image
    landmark = model_landmark(image)
    #reshape face landmark and convert to image coordinates
    landmark = landmark.reshape(68, 2)
    landmark[:,0] = landmark[:,0] * iExFaceW + iExFaceX
    landmark[:,1] = landmark[:,1] * iExFaceH + iExFaceY

    landmark = landmark.reshape(-1)

    return landmark


if __name__ == '__main__':
    dummy_input = torch.randn(1, 1, 64, 64)
    torch.onnx.export(model_landmark, dummy_input, "landmark.onnx", verbose=True, input_names=['input'], output_names=['output'])
