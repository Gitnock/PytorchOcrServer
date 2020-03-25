import multiprocessing
from multiprocessing import Pipe


import sys
import os
import time
import argparse




import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import pytesseract

from PIL import Image

import mss
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
from grab import snap_screen
from craft import CRAFT

from collections import OrderedDict

# title of our window
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()

# get screen size
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]

#
#   Size (in pixels) of the screen capture box to feed the neural net.
#   This box is in the center of your screen. Lower value makes the network faster.
#
#   Example: "ACTIVATION_RANGE = 400" means a 400x400 pixel box.
#
ACTIVATION_RANGE = 200

origbox = (int(Wd/2 - ACTIVATION_RANGE/2),
            int(Hd/2 - ACTIVATION_RANGE/2),
            int(Wd/2 + ACTIVATION_RANGE/2),
            int(Hd/2 + ACTIVATION_RANGE/2))

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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
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

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def grab_screen(p_input):
  while True:
    #Grab screen image
    img = np.array(snap_screen(region=origbox))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Put image from pipe
    p_input.send(img)

def find_text(p_output, p_input2):

    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

        # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True


    # load the input image and grab the image dimensions
    while True:

        # give image shape = square
        image = p_output.recv()
        
        # model output
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        
                


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
                

        p_input2.send(img)


def bounding_box(points):
            points = points.astype(np.int16)
            x_coordinates, y_coordinates = zip(*points)
            return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]
    

    

def Show_image(p_output2):
  print("[INFO] loading CRAFT text detector...")
  global start_time, fps
  while True:
    image_np = p_output2.recv()
    # Show image with detection
    cv2.imshow(title, image_np)
    # Bellow we calculate our FPS
    fps+=1
    TIME = time.time() - start_time
    if (TIME) >= display_time :
    #   print("FPS: ", fps / (TIME))
      fps = 0
      start_time = time.time()
    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
      sct.close()
      cv2.destroyAllWindows()
      sys.exit(0)
      break


    
if __name__=="__main__":
    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()
    # creating new processes
    p1 = multiprocessing.Process(target=grab_screen, args=(p_input,))
    p2 = multiprocessing.Process(target=find_text, args=(p_output,p_input2,))
    p3 = multiprocessing.Process(target=Show_image, args=(p_output2,))

    # starting our processes
    p1.start()
    p2.start()
    p3.start()