
import os
import sys

import argparse
from collections import OrderedDict
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn import DataParallel
from face_detect.test import Detection
from landmark_api import get_face_landmark
from face_feature.model import mobilefacenet, cbam, resnet
from face_utils.align_faces import warp_and_crop_face, get_reference_facial_points


def convert_68pts_5pts(landmark):
    left_eye_x = (landmark[74] + landmark[76] + landmark[80] + landmark[82]) / 4
    left_eye_y = (landmark[75] + landmark[77] + landmark[81] + landmark[83]) / 4

    right_eye_x = (landmark[86] + landmark[88] + landmark[92] + landmark[94]) / 4
    right_eye_y = (landmark[87] + landmark[89] + landmark[93] + landmark[95]) / 4

    nose_x, nose_y = landmark[60], landmark[61]

    left_mouse_x = (landmark[96] + landmark[120]) / 2
    left_mouse_y = (landmark[97] + landmark[121]) / 2

    right_mouse_x = (landmark[108] + landmark[128]) / 2
    right_mouse_y = (landmark[109] + landmark[129]) / 2
    return np.array([left_eye_x, right_eye_x, nose_x, left_mouse_x, right_mouse_x,
                     left_eye_y, right_eye_y, nose_y, left_mouse_y, right_mouse_y])


def align(f_bbox, img, output_size):

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmark = get_face_landmark(gray_image, f_bbox)

    facial5points = convert_68pts_5pts(landmark.detach().cpu().numpy())
    facial5points = np.reshape(facial5points, (2, 5))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial 5points, reference_5pts, crop_size)
    dst_img = warp_and_crop_face(img, facial5points, reference_pts=reference_5pts, crop_size=output_size)
    # cv2.imwrite(f'{0}_aligned_{output_size[0]}x{output_size[1]}.jpg', dst_img)
    # img = cv.resize(raw, (224, 224))
    # cv.imwrite('images/{}_img.jpg'.format(i), img)
    return dst_img


def load_model(backbone_net, feature_dim, gpus='0'):
    """ load the pretrained model """
    if backbone_net == 'MobileFace':
        net = mobilefacenet.MobileFaceNet(feature_dim=feature_dim)
    elif backbone_net == 'Res50':
        net = resnet.ResNet50()
    elif backbone_net == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=feature_dim, mode='ir')
    elif backbone_net == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=feature_dim, mode='ir')
    elif backbone_net == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=feature_dim, mode='ir_se')
    else:
        print(backbone_net, ' is not available!')

    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resume_path = os.path.join(os.path.dirname(__file__), 'face_feature/checkpoints/feature.ckpt')

    net.load_state_dict(torch.load(resume_path, map_location=device)['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    # convert model to onnx version

    # net.eval()
    # onnx_model_path = resume_path.replace('ckpt', 'onnx')
    # dummy_input = torch.randn(1, 3, 112, 112).to(device)
    # torch.onnx.export(net, dummy_input, onnx_model_path, verbose=True, input_names=["input"], output_names=["output"])

    return net.eval(), device


def get_feature(face_image, net, device):
    """ extract the feature vector from the input image """
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    face_img = transform(face_image).to(device)
    feature_vec = net(face_img.unsqueeze(0)).data.cpu().numpy()
    return feature_vec


def match_feature(feature1, feature2):

    mu = np.mean(np.concatenate((feature1, feature2), axis=0))
    feature1 = feature1 - mu
    feature2 = feature2 - mu

    feature1 = feature1 / np.expand_dims(np.sqrt(np.sum(np.power(feature1, 2), 1)), 1)
    feature2 = feature2 / np.expand_dims(np.sqrt(np.sum(np.power(feature2, 2), 1)), 1)

    score = np.sum(np.multiply(feature1, feature2), 1)
    return score


def get_feature_from_origin(image, feature_type):
    net, device = load_model('Res50', 512, '0')

    face_image = align(image, output_size=(112, 112))

    if feature_type == 0:
        feat_vec = get_feature(face_image, net, device)
    else:
        feat_vec = None

    return feat_vec


def get_feature_from_image(feature_extractor, bbox, device, image):

    face_image = align(bbox, image, output_size=(112, 112))
    if face_image is None:
        return None

    feat_vec = get_feature(face_image, feature_extractor, device)

    return feat_vec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer the model for face recognition')
    parser.add_argument('--backbone_net', type=str, default='Res50', help='MobileFace, Res50, CBAM_50, CBAM_50_SE, CBAM_100, CBAM_100_SE')
    parser.add_argument('--output_name', type=str, default='output', help='intermediate layer name of onnx model')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
    parser.add_argument('--feature_type', type=int, default=0, help='feature type - {0: float32, 1: int8}')
    parser.add_argument('--resume', type=str, default='./face_feature/checkpoints/feature.pth',
                        help='The path pf save checkpoints')
    parser.add_argument('--image', type=str, default='3.jpeg', help='source image file')
    parser.add_argument('--gpus', type=str, default='0', help='gpu list')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    print(get_feature_from_origin(image, args.feature_type))
