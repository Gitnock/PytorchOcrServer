import io
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms

import mss
import cv2
import pytesseract
# from skimage import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request , send_file

import imgproc
import craft_utils
from craft import CRAFT

from collections import OrderedDict

app = Flask(__name__)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

 # load net
net = CRAFT()     # initialize
net.load_state_dict(copyStateDict(torch.load("craft_mlt_25k.pth")))
net = net.cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = False
net.eval()       

@app.route('/')
def hello():
    return 'Wecome to the Watcher api'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        
        image = Image.open(io.BytesIO(img_bytes))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # image = image.recv()
        
        # model output
        bboxes, polys = get_prediction(net, image, 0.7, 0.4, 0.4, True, False, None)

        
                


        #render boxs on detected text
        img = np.array(image)
        verticals=None
        texts=None
        for i, box in enumerate(polys):
                poly = np.array(box).astype(np.int32).reshape((-1))

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)



        for i, bbs in enumerate(bboxes):
            box = bounding_box(bbs)
            roi = image[box[0][1]:box[1][1],box[0][0]:box[1][0]]

            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(roi, config=config)

            print(text)
            # Close when found ACCEPT
            # if text == "ACCEPT":
            #       print(text)
        


        # OLD ver THAT CROPS IMAGE
        # #read text
        # for i, bbs in enumerate(bboxes):
        #         crop = bounding_box(bbs)
        #         cropped = image[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]
        #         print(pytesseract.image_to_string(cropped))
                

        # p_input2.send(img)




        return send_file(io.BytesIO(img), mimetype='image/jpg')



def get_prediction(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]


    return boxes, polys



def bounding_box(points):
            points = points.astype(np.int16)
            x_coordinates, y_coordinates = zip(*points)
            return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

if __name__ == '__main__':
    app.debug = True
    app.run(host = '127.0.0.1',port=5000)