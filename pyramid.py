import os
import cv2
import argparse
import numpy as np
from lukas_kanade import *
import lukas_kanade as lk

template = None
template_list = None
rows = None
cols = None

def generate_Pyramid(img1):
    src = img1
    pyramid = [src]
    for i in range(0,3):
        src = cv2.pyrDown(src)
        pyramid.append(src)
        
    return pyramid

def generate_coordinate(coord):
    new_coord = coord
    coordinate_list = [coord]
    for i in range(0,3):
        new_coord = (new_coord[0]//2,new_coord[1]//2)
        coordinate_list.append(new_coord)
        
    return coordinate_list
        
    
def translation_Pyramid(img_template, img_search, coordinate, dimension) :   
    
    initial_parameters = np.array([0,0])
    for i in range(2,-1,-1):
        initial_parameters = 2* initial_parameters
        lk.template = template_list[i]
        lk.rows, lk.cols = lk.template.shape
        initial_parameters = translation_tracker(img_template, img_search[i], initial_parameters, coordinate[i], (template_list[i].shape[1],template_list[i].shape[0]),max_iter=500, learning_rate=1)
    return initial_parameters
        
def affine_Pyramid(img_template, img_search, coordinate, dimension) :   
    
    initial_parameters = np.array([0,0,0,0,0,0])
    for i in range(2,-1,-1):
        initial_parameters[4] = 2* initial_parameters[4]
        initial_parameters[5] = 2* initial_parameters[5]
        lk.template = template_list[i]
        lk.rows, lk.cols = lk.template.shape
        initial_parameters = affine_tracker(img_template, img_search[i], initial_parameters, coordinate[i], (template_list[i].shape[1],template_list[i].shape[0]),max_iter=100, learning_rate=1)
    return initial_parameters
        


        
        
file1= open('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/BlurCar2/groundtruth_rect.txt')
Lines = file1.readlines()
rectangles = []
for line in Lines:
    word= line.split()
    xyz = []
    xyz.append((int(word[0]),(int(word[1]))))
    xyz.append((int(word[2]),(int(word[3]))))
    rectangles.append(xyz)

img2 = cv2.imread('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/BlurCar2/img/0001.jpg')
initial_rect = rectangles[0][0]
initial_parameters = np.array([0,0])
dimensions = rectangles[0][1]
dimensions_list= generate_coordinate(dimensions)
initial_rect_list = generate_coordinate(initial_rect)
end_point = (rectangles[0][0][0]+rectangles[0][1][0],rectangles[0][0][1]+rectangles[0][1][1])
ghi = cv2.rectangle(img2, rectangles[0][0] , end_point, (255, 0, 0), 2)
cv2.imwrite('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/BlurCar2/output/0001.jpg',ghi)
   
template = crop(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),initial_rect, dimensions)
rows, cols = template.shape
lk.rows, lk.cols = template.shape
lk.template = template
template_list = generate_Pyramid(template)
cv2.imwrite('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/BlurCar2/output/template.jpg',template)

mask_iou_store = []
for i in range(2,585):
    print(i)
    img1 = img2
    img2 = cv2.imread('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/BlurCar2/img/'+ str(i).zfill(4)+'.jpg')
    store_img_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    store_img_2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)   
    
    store_img_2_list = generate_Pyramid(store_img_2)
    

    parameters = affine_Pyramid(store_img_1,store_img_2_list,initial_rect_list,dimensions_list)
    initial_parameters = parameters
    point_list, ghi = affine_plotter(img2,rectangles[0][0], rectangles[0][1],parameters)
    for  j in range(0,len(point_list)):
       point_list[j] = [point_list[j][0],point_list[j][1]]
    point_list = np.array(point_list)
    mask_img = mask_plotter(point_list,img2)
    mask_store = np.zeros((img2.shape[0],img2.shape[1]))
    mask_store[rectangles[i][0][1]:rectangles[i][0][1]+rectangles[i][1][1],rectangles[i][0][0]:rectangles[i][0][0]+rectangles[i][1][0]] = 255
    val = binary_mask_iou(mask_img,mask_store)
    mask_iou_store.append(val)    
    cv2.imwrite('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/BlurCar2/output/'+str(i).zfill(4)+'.jpg',ghi)
     

mask_iou_store = np.array(mask_iou_store)
print(np.mean(mask_iou_store))
