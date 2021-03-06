from pyramid import *
from lukas_kanade import *
from ebma import *
import sys
import cv2
import numpy as np
import pyramid as pyr
import lukas_kanade as lk

if int(sys.argv[1]) == 1:
    
    bboxes = []
    with open(sys.argv[2]) as gt:
        all_lines = gt.readlines()
        for line in all_lines:
            nums = line.split(',')
            bboxes.append((int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])))

    pic_count = len(bboxes)
    
    first_frame = cv2.imread(sys.argv[3]+'0001.jpg')
    first_frame = cv2.cvtColor(first_frame,cv2.COLOR_RGB2GRAY)
    ebma = BlockMatch(first_frame[bboxes[0][1]:bboxes[0][1]+bboxes[0][3],bboxes[0][0]:bboxes[0][0]+bboxes[0][2]], bboxes[0], alpha=0.0)
    
    IOUs = []
    for i in range(2, pic_count+1):
        frame = cv2.imread(sys.argv[3] + str(i).zfill(4) + '.jpg')
        frame_bw = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        best_match = ebma.get_best_match(frame_bw, 5)
        IOUs.append(iou(best_match, bboxes[i-1]))
        tracked_frame = cv2.rectangle(frame, (best_match[0], best_match[1]), (best_match[0]+best_match[2], best_match[1]+best_match[3]), (255, 0, 0), 2)
        cv2.imwrite(sys.argv[4] + str(i).zfill(4) + '.jpg', tracked_frame)
    print('mIOU = ' + str(np.mean(IOUs)))
    
    
if int(sys.argv[1])==2:
 
    point_list = None
    file1= open(sys.argv[2])
    Lines = file1.readlines()
    rectangles = []
    for line in Lines:
        word= line.split('\t')
        xyz = []
        xyz.append((int(word[0]),(int(word[1]))))
        xyz.append((int(word[2]),(int(word[3]))))
        rectangles.append(xyz)
    pic_count = len(rectangles)
    img2 = cv2.imread(sys.argv[3]+'0001.jpg')
    initial_rect = rectangles[0][0]
    initial_parameters = np.array([0,0,0,0,0,0])
    dimensions = rectangles[0][1]
    end_point = (rectangles[0][0][0]+rectangles[0][1][0],rectangles[0][0][1]+rectangles[0][1][1])
    ghi = cv2.rectangle(img2, rectangles[0][0] , end_point, (255, 0, 0), 2)
    cv2.imwrite(sys.argv[4]+'0001.jpg',ghi)
    template = crop(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),initial_rect, dimensions)
    rows, cols = template.shape
    lk.rows, lk.cols = template.shape
    lk.template = template
    mask_iou_store = []
    for i in range(2,pic_count):
        img1 = img2
        img2 = cv2.imread(sys.argv[3]+ str(i).zfill(4)+'.jpg')
        store_img_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        store_img_2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
        
        if sys.argv[5]== '1':
            parameters = translation_tracker(store_img_1,store_img_2, np.array([0,0]),initial_rect,dimensions,max_iter=100,learning_rate=1)
            point_list, ghi = translation_plotter(img2,rectangles[0][0], rectangles[0][1],parameters)
            
        elif sys.argv[5]== '2':
            parameters = affine_tracker(store_img_1,store_img_2, np.array([0,0,0,0,0,0]),initial_rect,dimensions,max_iter=100,learning_rate=1)
            point_list, ghi = affine_plotter(img2,rectangles[0][0], rectangles[0][1],parameters)
             
        elif sys.argv[5]== '3':
            parameters = projective_tracker(store_img_1,store_img_2, np.array([0,0,0,0,0,0,0,0]),initial_rect,dimensions,max_iter=100,learning_rate=1)
            point_list, ghi = projective_plotter(img2,rectangles[0][0], rectangles[0][1],parameters)
            
        else:
            print("Incorrect option.")
            
        for j in range(0,len(point_list)):
            point_list[j] = [point_list[j][0],point_list[j][1]]
            point_list = np.array(point_list)
       
        mask_img = mask_plotter(point_list,img2)
        mask_store = np.zeros((img2.shape[0],img2.shape[1]))
        mask_store[rectangles[i][0][1]:rectangles[i][0][1]+rectangles[i][1][1],rectangles[i][0][0]:rectangles[i][0][0]+rectangles[i][1][0]] = 255
        val = binary_mask_iou(mask_img,mask_store)
        mask_iou_store.append(val)    
        cv2.imwrite(sys.argv[4]+str(i).zfill(4)+'.jpg',ghi)
    mask_iou_store = np.array(mask_iou_store)
    print(np.mean(mask_iou_store))
         
         
elif int(sys.argv[1])==3:
    
    file1= open(sys.argv[2])
    Lines = file1.readlines()
    rectangles = []
    for line in Lines:
        word= line.split('\t')
        xyz = []
        xyz.append((int(word[0]),(int(word[1]))))
        xyz.append((int(word[2]),(int(word[3]))))
        rectangles.append(xyz)
        
        
    img2 = cv2.imread(sys.argv[3]+'0001.jpg')
    initial_rect = rectangles[0][0]
    initial_parameters = np.array([0,0,0,0,0,0])
    dimensions = rectangles[0][1]
    dimensions_list= generate_coordinate(dimensions)
    initial_rect_list = generate_coordinate(initial_rect)
    end_point = (rectangles[0][0][0]+rectangles[0][1][0],rectangles[0][0][1]+rectangles[0][1][1])
    ghi = cv2.rectangle(img2, rectangles[0][0] , end_point, (255, 0, 0), 2)
    cv2.imwrite(sys.argv[4]+'0001.jpg',ghi)
    
    pyr.template = crop(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),initial_rect, dimensions)
    pyr.rows, pyr.cols = pyr.template.shape
    pyr.template_list = generate_Pyramid(pyr.template)
    mask_iou_store = []
    pic_count = len(rectangles)
    for i in range(2,pic_count):
        
        img1 = img2
        img2 = cv2.imread(sys.argv[3]+ str(i).zfill(4)+'.jpg')
        store_img_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        store_img_2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)   
        
        store_img_2_list = generate_Pyramid(store_img_2)
        if sys.argv[5]== '1' :
            parameters = translation_Pyramid(store_img_1,store_img_2_list,initial_rect_list,dimensions_list)
            point_list, ghi = translation_plotter(img2,rectangles[0][0], rectangles[0][1],parameters)
        elif sys.argv[5]== '2' :
            parameters = affine_Pyramid(store_img_1,store_img_2_list,initial_rect_list,dimensions_list)
            point_list, ghi = affine_plotter(img2,rectangles[0][0], rectangles[0][1],parameters)
        else:
            parameters = projective_Pyramid(store_img_1,store_img_2_list,initial_rect_list,dimensions_list)
            point_list, ghi = projective_plotter(img2,rectangles[0][0], rectangles[0][1],parameters)
        
        for  j in range(0,len(point_list)):
            point_list[j] = [point_list[j][0],point_list[j][1]]
        point_list = np.array(point_list)
        mask_img = mask_plotter(point_list,img2)
        mask_store = np.zeros((img2.shape[0],img2.shape[1]))
        mask_store[rectangles[i][0][1]:rectangles[i][0][1]+rectangles[i][1][1],rectangles[i][0][0]:rectangles[i][0][0]+rectangles[i][1][0]] = 255
        val = binary_mask_iou(mask_img,mask_store)
        mask_iou_store.append(val)    
        cv2.imwrite(sys.argv[4]+str(i).zfill(4)+'.jpg',ghi)
    mask_iou_store = np.array(mask_iou_store)
    print(np.mean(mask_iou_store))
    
    
