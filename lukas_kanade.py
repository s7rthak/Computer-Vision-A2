import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

template = None
rows = None
cols = None

def binary_mask_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1==255,  mask2==255))
    union = mask1_area+mask2_area-intersection
    if union == 0: 
        # only happens if both masks are background with all zero values
        iou = 1
    else:
        iou = intersection/union 
    return iou



#returns the required gradient matrix


def crop(image,coordinate,dimension):
    return image[(int)(coordinate[1]):(int)(coordinate[1]+dimension[1]),int(coordinate[0]):int(coordinate[0]+dimension[0])]


#Returns corresponding Jacobian
def jacobian_translation(x_shape,y_shape,coordinates):
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))
    row1 = np.stack((ones,zeros), axis=2)
    row2 = np.stack((zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)
    return jacob





def translation_tracker(img_template, img_search, initial_parameters, coordinate, dimension, max_iter=500,learning_rate=0.05):
    parameters = np.array([0,0])
    parameters = initial_parameters
    iterations = 0
    
    while(iterations<max_iter):
        #General preprocessing
        iterations += 1     
        new_coordinate = (coordinate[0]+parameters[0],coordinate[1]+parameters[1])
        store_img = img_search
        temp_search = crop(store_img,new_coordinate,dimension)
        
        #First obtaining respective gradient as per warp
        dx = cv2.Sobel(img_search,cv2.CV_64F,1,0,ksize=5)
        gradient_dx = crop(dx,new_coordinate,dimension)
        dy = cv2.Sobel(img_search,cv2.CV_64F,0,1,ksize=5)
        gradient_dy = crop(dy,new_coordinate,dimension)
        gradient_fin = np.stack((gradient_dx,gradient_dy),axis=2)
        gradient_fin = np.expand_dims(gradient_fin,axis=2)
        
        
        #Applying Lukas Kanade Formula
        jacob = jacobian_translation(cols,rows,coordinate)
        steepest_descent = np.matmul(gradient_fin,jacob)
        steepest_descent_T = np.rollaxis(steepest_descent,3,2)
        hessian = np.matmul(steepest_descent_T,steepest_descent).sum((0,1))
        hess_inv = np.linalg.pinv(hessian)
        
        #Difference accumulate
        diff = template.astype(int) - temp_search.astype(int)
        diff = diff.reshape((rows,cols,1,1))
        update = (steepest_descent_T * diff).sum((0,1))
        dp = np.matmul(hess_inv,update).reshape(-1)
       
        #Needs to be checked, not sure about the +- thing @Sarthak
        parameters =parameters + learning_rate * dp
        
        norm = np.linalg.norm(dp)
        if(norm) < 0.001:
            break
        
    return parameters



#Returns corresponding Jacobian
def jacobian_affine(x_shape,y_shape,coordinates):
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x,y = np.meshgrid(x, y)
    x = x + coordinates[1]
    y = y + coordinates[0]
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))
    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)
    return jacob

#Input image can have some associated affine transformation as well 
def affine_tracker(img_template, img_search, initial_parameters, coordinate, dimension, max_iter=100,learning_rate=0.05):
    
    global template
    
    parameters = np.array([0,0,0,0,0,0])
    parameters = initial_parameters
    iterations = 0
    
    while(iterations<max_iter):
        #General preprocessing
        iterations += 1     
        warp_mat_2 = np.array([[1+parameters[0],parameters[2],parameters[4]],[parameters[1],1+parameters[3],parameters[5]]] )
        warp_mat_2 = warp_mat_2.astype(np.float32)
        store_img = cv2.warpAffine(img_search,warp_mat_2,(img_template.shape[1],img_template.shape[0]),flags=cv2.INTER_CUBIC)
        temp_search = crop(store_img,coordinate,dimension)
        
        #First obtaining respective gradient as per warp
        dx = cv2.Sobel(img_search,cv2.CV_64F,1,0,ksize=5)
        gradient_dx = cv2.warpAffine(dx,warp_mat_2,(img_search.shape[1],img_search.shape[0]),flags=cv2.INTER_CUBIC)
        gradient_dx = crop(gradient_dx,coordinate,dimension)
        dy = cv2.Sobel(img_search,cv2.CV_64F,0,1,ksize=5)
        gradient_dy = cv2.warpAffine(dy,warp_mat_2,(img_search.shape[1],img_search.shape[0]),flags=cv2.INTER_CUBIC)
        gradient_dy = crop(gradient_dy,coordinate,dimension)
        gradient_fin = np.stack((gradient_dx,gradient_dy),axis=2)
        gradient_fin = np.expand_dims(gradient_fin,axis=2)
        
        #Applying Lukas Kanade Formula
        jacob = jacobian_affine(cols,rows,coordinate)
        steepest_descent = np.matmul(gradient_fin,jacob)
        steepest_descent_T = np.rollaxis(steepest_descent,3,2)
        hessian = np.matmul(steepest_descent_T,steepest_descent).sum((0,1))
        hess_inv = np.linalg.pinv(hessian)
        
        #Difference accumulate
        diff = template.astype(int) - temp_search.astype(int)
        diff = diff.reshape((rows,cols,1,1))
        update = (steepest_descent_T * diff).sum((0,1))
        dp = np.matmul(hess_inv,update).reshape(-1)
       
        #Needs to be checked, not sure about the +- thing @Sarthak
        parameters =parameters + learning_rate * dp
        
        norm = np.linalg.norm(dp)
        if(norm) < 0.001:
            break
        
    return parameters
        

def jacobian_projection(parameters,x_shape,y_shape,coordinates):
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x,y = np.meshgrid(x, y)
    x += coordinates[1]
    y += coordinates[0]
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))
    
    num_x = (1+parameters[0])*x + (parameters[3])*y + parameters[6]
    num_y = (parameters[1])*x + (1+parameters[4])*y + parameters[7]
    den = (parameters[2])*x + (parameters[5])*y + 1
    row1 = np.stack((x*den, zeros, -1*x*num_x, y*den, zeros, -1*y*num_x, den, zeros), axis=2)
    row2 = np.stack((zeros, x*den, -1*x*num_y, zeros, y*den, -1*y*num_y, zeros, den), axis=2)
    jacob = np.stack((row1, row2), axis=2)
    jacob = jacob/ (den * den).reshape(y_shape,x_shape,1,1)
    return jacob
        
def projective_tracker(img_template, img_search, initial_parameters ,coordinate, dimension, max_iter=100,learning_rate=0.001):
    parameters = np.array([0,0,0,0,0,0,0,0])
    parameters = initial_parameters
    iterations = 0
    
    while(iterations<max_iter):
        #General preprocessing
        iterations += 1     
        warp_mat_2 = np.array([[1+ parameters[0], parameters[3], parameters[6]],[parameters[1],1+parameters[4], parameters[7]],[parameters[2], parameters[5], 1]])
        warp_mat_2 = warp_mat_2.astype(np.float32)
        store_img = cv2.warpPerspective(img_search,warp_mat_2,(img_template.shape[1],img_template.shape[0]),flags=cv2.INTER_CUBIC)
        temp_search = crop(store_img,coordinate,dimension)
        
        #First obtaining respective gradient as per warp
        dx = cv2.Sobel(img_search,cv2.CV_64F,1,0,ksize=5)
        gradient_dx = cv2.warpPerspective(dx,warp_mat_2,(img_search.shape[1],img_search.shape[0]),flags=cv2.INTER_CUBIC)
        gradient_dx = crop(gradient_dx,coordinate,dimension)
        dy = cv2.Sobel(img_search,cv2.CV_64F,0,1,ksize=5)
        gradient_dy = cv2.warpPerspective(dy,warp_mat_2,(img_search.shape[1],img_search.shape[0]),flags=cv2.INTER_CUBIC)
        gradient_dy = crop(gradient_dy,coordinate,dimension)
        gradient_fin = np.stack((gradient_dx,gradient_dy),axis=2)
        gradient_fin = np.expand_dims(gradient_fin,axis=2)
        
        #Applying Lukas Kanade Formula
        jacob = jacobian_projection(parameters,cols,rows,coordinate)
        steepest_descent = np.matmul(gradient_fin,jacob)
        steepest_descent_T = np.rollaxis(steepest_descent,3,2)
        hessian = np.matmul(steepest_descent_T,steepest_descent).sum((0,1))
        hess_inv = np.linalg.pinv(hessian)
        
        #Difference accumulate
        diff = template.astype(int) - temp_search.astype(int)
        diff = diff.reshape((rows,cols,1,1))
        update = (steepest_descent_T * diff).sum((0,1))
        dp = np.matmul(hess_inv,update).reshape(-1)
       
        #Needs to be checked, not sure about the +- thing @Sarthak
        parameters =parameters + learning_rate * dp
        
        norm = np.linalg.norm(dp)
        if(norm) < 0.001:
            break
        
    return parameters


def translation_mapper(point,parameters):
    new_coordinate_x = (int)(parameters[0]+point[0])
    new_coordinate_y = (int)(parameters[1]+point[1])
    return (new_coordinate_x,new_coordinate_y)

def translation_plotter(img,point,dimension,parameters):
    point_list = []
    point_list.append((point[0],point[1]))
    point_list.append((point[0]+ dimension[0],point[1]))
    point_list.append((point[0]+ dimension[0],point[1]+ dimension[1]))
    point_list.append((point[0],point[1]+ dimension[1]))
    for i in range(0,4):
        point_list[i] = translation_mapper(point_list[i],parameters)
    img1 = img
    img1= cv2.line(img1,point_list[0],point_list[1],(255,0,0),2)
    img1= cv2.line(img1,point_list[1],point_list[2],(255,0,0),2)
    img1= cv2.line(img1,point_list[2],point_list[3],(255,0,0),2)
    img1= cv2.line(img1,point_list[3],point_list[0],(255,0,0),2)
    return point_list, img1


def affine_mapper(point,parameters):
    new_coordinate_x = (int)(((1+parameters[0])*point[0])+ (parameters[2]* point[1]) + parameters[4])
    new_coordinate_y = (int)((parameters[1]*point[0])+ ((1+parameters[3])* point[1]) + parameters[5])
    return (new_coordinate_x,new_coordinate_y)

def affine_plotter(img,point,dimension,parameters):
    point_list = []
    point_list.append((point[0],point[1]))
    point_list.append((point[0]+ dimension[0],point[1]))
    point_list.append((point[0]+ dimension[0],point[1]+ dimension[1]))
    point_list.append((point[0],point[1]+ dimension[1]))
    for i in range(0,4):
        point_list[i] = affine_mapper(point_list[i],parameters)
    img1 = img
    img1= cv2.line(img1,point_list[0],point_list[1],(255,0,0),2)
    img1= cv2.line(img1,point_list[1],point_list[2],(255,0,0),2)
    img1= cv2.line(img1,point_list[2],point_list[3],(255,0,0),2)
    img1= cv2.line(img1,point_list[3],point_list[0],(255,0,0),2)
    return point_list, img1


def projective_mapper(point,parameters):
    new_coordinate_x = (int)((((1+parameters[0])*point[0])+ (parameters[3]* point[1]) + parameters[6])/((parameters[2]*point[0])+ (parameters[5]* point[1]) + 1))
    new_coordinate_y = (int)(((parameters[1]*point[0])+ ((1+parameters[3])* point[1]) + parameters[7])//((parameters[2]*point[0])+ (parameters[5]* point[1]) + 1))
    return (new_coordinate_x,new_coordinate_y)

def projective_plotter(img,point,dimension,parameters):
    point_list = []
    point_list.append((point[0],point[1]))
    point_list.append((point[0]+ dimension[0],point[1]))
    point_list.append((point[0]+ dimension[0],point[1]+ dimension[1]))
    point_list.append((point[0],point[1]+ dimension[1]))
    for i in range(0,4):
        point_list[i] = projective_mapper(point_list[i],parameters)
    img1 = img
    img1= cv2.line(img1,point_list[0],point_list[1],(255,0,0),2)
    img1= cv2.line(img1,point_list[1],point_list[2],(255,0,0),2)
    img1= cv2.line(img1,point_list[2],point_list[3],(255,0,0),2)
    img1= cv2.line(img1,point_list[3],point_list[0],(255,0,0),2)
    return point_list,img1


def mask_plotter(point_list,img):
    mask = np.zeros((img.shape[0],img.shape[1]))
    cv2.fillConvexPoly(mask,point_list,255)
    mask = mask.astype(int)
    return mask


    
