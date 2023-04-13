import os
import sys
import cv2
import math
import numpy as np


class Detection:
    def __init__(self):
        src_dir = os.path.dirname(__file__)
        if not os.path.exists(os.path.join(src_dir, "checkpoints")):
            os.makedirs(os.path.join(src_dir, "checkpoints"))

        caffemodel = os.path.join(src_dir, "checkpoints/Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(src_dir, "checkpoints/deploy.prototxt")

        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()

        bboxes = []
        for i in range(out.shape[0]):
            # conf_index = np.argmax(out[:, 2])
            if out[i][2] < self.detector_confidence:
                continue

            left, top, right, bottom = out[i, 3]*width, out[i, 4]*height, out[i, 5]*width, out[i, 6]*height

            if right == left or bottom == top:
                continue

            bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
            bboxes.append(bbox)

        return bboxes

    def check_face(self):
        pass


if __name__ == '__main__':

    src_dir = 'D:/19.Database/office_angled_db'
    dst_dir = 'D:/19.Database/office_angled_db_result'
    detector = Detection()

    for file in os.listdir(src_dir):
        image1 = cv2.imread(os.path.join(src_dir, file))
        box = detector.get_bbox(image1)
        if box:
            cv2.rectangle(image1, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 5)

        cv2.imwrite(os.path.join(dst_dir, file), image1)
        # cv2.waitKey(0)
