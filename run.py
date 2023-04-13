import sys
import os
import random
import cv2
from face_detect.test import Detection, get_bbox
from feature_api import load_model, get_feature_from_image, match_feature
from tqdm import tqdm
detector = Detection()
net, device = load_model('Res50', 512, '0')


def get_file_names(dir_path):
    file_list = os.listdir(dir_path)
    total_file_list = []

    for node in file_list:
        full_path = os.path.join(dir_path, node)
        if os.path.isdir(full_path):
            total_file_list = total_file_list + get_file_names(full_path)
        else:
            _, file_extension = os.path.splitext(full_path)
            if file_extension in ['.jpg', '.png', '.bmp', '.mp4', '.avi']:
                total_file_list.append(full_path)

    return total_file_list


def show_image(cam, text, color=(0, 0, 255)):
    """ show image and print message """
    _, image = cam.read()
    if image is None:
        return None

    # image = imutils.resize(image, width=720)
    # im = cv2.flip(im, 1)
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    return image


def enroll_images(src_dir):
    features = {}
    all_images = get_file_names(src_dir)
    for path in tqdm(all_images):
        image = cv2.imread(path)
        bboxes = get_bbox(image)
        for bbox in bboxes:
            feature_vec = get_feature_from_image(net, bbox, device, image)

            if feature_vec is None:
                continue

            label = os.path.basename(os.path.dirname(path))

            if label not in features.keys():
                features[label] = [feature_vec]
            else:
                features[label].append(feature_vec)

    return features


def main():
    enroll_dir = './datasets/Enroll'
    features = enroll_images(enroll_dir)

    cv2.namedWindow("Face Recognition Demo")
    cap = cv2.VideoCapture("./datasets/office.mp4")
    while not cap.isOpened():
        cap = cv2.VideoCapture("./out.mp4")
        cv2.waitKey(1000)
        print("Wait for the header")

    while True:
        flag, frame = cap.read()
        show_img = frame.copy()
        bboxes = get_bbox(frame)
        for bbox in bboxes:
            feat2 = get_feature_from_image(net, bbox, device, frame)

            if feat2 is None:
                continue

            max_score = 0
            max_label = None
            for label in features.keys():
                for feat1 in features[label]:
                    score = match_feature(feat1, feat2)
                    if score > max_score:
                        max_score = score
                        max_label = label

            cv2.putText(show_img, f'{max_label} {max_score[0]:.02f}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_COMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.rectangle(show_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 3)

        if flag:
            # The frame is ready and already captured
            cv2.imshow('Face Recognition Demo', show_img)
        else:
            # It is better to wait for a while for the next frame to be ready
            break

        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
