import pyramid as pyr
import lukas_kanade as lk
import ebma
import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Take template path')
parser.add_argument('-t', '--template', type=str, help='path to template', default='./template.jpg')

args = parser.parse_args()

USE_GUI = True

template_path = args.template
template_img = None
template_img_bw = None
bm = None
if not USE_GUI:
    template_img = cv2.imread(template_path)
    template_img_bw = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
initial_bbox = None
initial_rect = None
dimensions = None
initial_rect_list = None
dimensions_list = None
end_point = None
layers=7

video_object = cv2.VideoCapture(0)

prev_frame = None
frame_no = 1
while True:
    ret, frame = video_object.read()

    # Initialize some variables 
    if frame_no == 1:
        if USE_GUI:
            r = cv2.selectROI(frame)
            template_img = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            template_img_bw = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            initial_bbox = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        else:
            initial_bbox = bm.get_best_match(frame, stride=3)
        bm = ebma.BlockMatch(template_img, initial_bbox)
        print(initial_bbox)
        prev_frame = frame
    elif frame_no == 2:
        initial_rect = (initial_bbox[0], initial_bbox[1])
        dimensions = (initial_bbox[2], initial_bbox[3])
        dimensions_list= pyr.generate_coordinate(dimensions,no_layers=layers)
        initial_rect_list = pyr.generate_coordinate(initial_rect,no_layers=layers)
        end_point = (initial_rect[0]+dimensions[0], initial_rect[1]+dimensions[1])

        pyr.template = pyr.template = lk.crop(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),initial_rect, dimensions)
        pyr.rows, pyr.cols = pyr.template.shape
        pyr.template_list = pyr.generate_Pyramid(pyr.template,no_layers=layers)
        lk.template = pyr.template
        lk.rows, lk.cols = pyr.template.shape
    
    if frame_no > 1:
        store_img_1 = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
        store_img_2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   
        prev_frame = frame

        ################# Block Match #################
        best_match = bm.get_best_match(frame, 10, stride=3)
        ghi = cv2.rectangle(frame, (best_match[0], best_match[1]), (best_match[0]+best_match[2], best_match[1]+best_match[3]), (255, 0, 0), 2)

        ################# Without pyramid ################
        # parameters = lk.affine_tracker(store_img_1,store_img_2, np.array([0,0,0,0,0,0]),initial_rect,dimensions,max_iter=20,learning_rate=1)
        # point_list, ghi = lk.affine_plotter(frame,initial_rect,dimensions,parameters)

        ################# Pyramid ##################
        # store_img_2_list = pyr.generate_Pyramid(store_img_2,no_layers=layers)
        # parameters = pyr.affine_Pyramid(store_img_1,store_img_2_list,initial_rect_list,dimensions_list,no_layers=layers)
        # point_list, ghi = pyr.affine_plotter(frame,initial_rect,dimensions,parameters)
        
        cv2.imshow('frame', ghi)
        cv2.waitKey(1)

    frame_no += 1
    if 0xFF == ord('q'):
        break

video_object.release()
cv2.destroyAllWindows()
