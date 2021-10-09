import os
import cv2
import argparse
import numpy as np

def iou_compute(rect1, dim1,rect2, dim2):
    
    (x_left_1) = rect1[0]
    (x_right_1) = rect1[0] + dim1[0]
    (y_top_1) = rect1[1]
    (y_bottom_1) = rect1[1] + dim1[1]
    
    (x_left_2) = rect2[0]
    (x_right_2) = rect2[0] + dim2[0]
    (y_top_2) = rect2[1]
    (y_bottom_2) = rect2[1] + dim2[1]
    
    x_left_common = max(x_left_1,x_left_2)
    x_right_common = min(x_right_1,x_right_2)
    y_top_common = max(y_top_1,y_top_2)
    y_bottom_common = min(y_bottom_1,y_bottom_2)
    
    common_area = None
    
    if x_right_common < x_left_common or y_bottom_common < y_top_common:
        common_area = 0
    else:
        common_area = (x_right_common - x_left_common) *  (y_bottom_common - y_top_common)
    
    union_area = dim1[0]*dim1[1] + dim2[0]*dim2[1] - common_area
    return common_area/union_area

    

def block_search_function(img1,img2,corner,dimension,search_radius,stride,type_dist):
    
    min_difference = float('inf')
    coordinates = None
    store_img = img1[corner[0]:corner[0]+dimension[0]+1,corner[1]:corner[1]+dimension[1]+1]
    
    if(type_dist==2):
        store_img = (store_img - np.mean(store_img.flatten())/np.std(store_img.flatten()))
        min_difference = -1 * min_difference
    
    for i in range(max(0,corner[0]-search_radius),min(img1.shape[0],corner[0]+search_radius)+1,stride):
        for j in range(max(0,corner[1]-search_radius),min(img1.shape[1],corner[1]+search_radius)+1,stride):
            if ((i+dimension[0]) > min(img1.shape[0],corner[0]+search_radius) ) :
                continue
            if ((j+dimension[1]) > min(img1.shape[1],corner[1]+search_radius) ) :
                continue
            store_2 = img2[i:i+store_img.shape[0],j:j+store_img.shape[1]]
            
            if store_2.shape != store_img.shape:
                continue
            #Mean Squared distance
            if type_dist == 1:
                abc = (store_2 - store_img) ** 2
                val = np.sum(abc.flatten())
                if val < min_difference:
                    min_difference = val
                    coordinates = (i,j)
            else:
                abc = (store_2- np.mean(store_2))/np.std(store_2)
                val = np.sum((np.dot(store_img,abc.T)).flatten())
                if val > min_difference:
                    min_difference = val
                    coordinates = (i,j)
                
    return coordinates
                                
                


file1= open('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/Bolt/groundtruth_rect.txt')
Lines = file1.readlines()
rectangles = []
for line in Lines:
    word= line.split(',')
    xyz = []
    xyz.append((int(word[0]),(int(word[1]))))
    xyz.append((int(word[2]),(int(word[3]))))
    rectangles.append(xyz)
                   
    
print('Reading done.')
    
img2 = cv2.imread('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/Bolt/img/0001.jpg')
initial_rect = rectangles[0][0]
dimensions = rectangles[0][1]

end_point = (rectangles[0][0][0]+rectangles[0][1][0],rectangles[0][0][1]+rectangles[0][1][1])
ghi = cv2.rectangle(img2, rectangles[0][0] , end_point, (255, 0, 0), 2)
cv2.imwrite('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/BlurCar2/output/0001.jpg',ghi)

miou_tot = 0
for i in range(2,585):
    img1 = img2
    img2 = cv2.imread('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/Bolt/img/'+ str(i).zfill(4)+'.jpg')
    store_img_1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    store_img_2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)   
    coordinates = block_search_function(store_img_1,store_img_2,initial_rect,dimensions,max(dimensions[0],dimensions[1]),1,2)
    end_point = (coordinates[0]+dimensions[0],coordinates[1]+dimensions[1])
    ghi = cv2.rectangle(img2, coordinates , end_point, (255, 0, 0), 2)
    cv2.imwrite('/Users/shreyansnagori/Desktop/COL780 Assignment 2/A2/Bolt/output/ '+str(i).zfill(4)+'.jpg',ghi)
    miou_tot += iou_compute(rectangles[i-1][0],dimensions,coordinates,dimensions)
    initial_rect = coordinates

miou_tot = miou_tot/584
print(miou_tot)
    
