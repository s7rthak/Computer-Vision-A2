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


def generate_Pyramid(img1,no_layers):
    src = img1
    pyramid = [src]
    for i in range(0,no_layers-1):
        src = cv2.pyrDown(src)
        pyramid.append(src)
        
    return pyramid

def generate_coordinate(coord,no_layers):
    new_coord = coord
    coordinate_list = [coord]
    for i in range(0,no_layers-1):
        new_coord = (new_coord[0]//2,new_coord[1]//2)
        coordinate_list.append(new_coord)
        
    return coordinate_list
        
    
def translation_Pyramid(img_template, img_search, coordinate, dimension,no_layers,max_iter=100) :   
    
    initial_parameters = np.array([0,0])
    for i in range(no_layers-1,-1,-1):
        initial_parameters = 2* initial_parameters
        lk.template = template_list[i]
        lk.rows, lk.cols = lk.template.shape
        initial_parameters = translation_tracker(img_template, img_search[i], initial_parameters, coordinate[i], (template_list[i].shape[1],template_list[i].shape[0]),max_iter=500, learning_rate=1)
    return initial_parameters
        
def affine_Pyramid(img_template, img_search, coordinate, dimension, no_layers,max_iter=100) :   
    
    initial_parameters = np.array([0,0,0,0,0,0])
    for i in range(no_layers-1,-1,-1):
        initial_parameters[4] = 2* initial_parameters[4]
        initial_parameters[5] = 2* initial_parameters[5]
        lk.template = template_list[i]
        lk.rows, lk.cols = lk.template.shape
        initial_parameters = affine_tracker(img_template, img_search[i], initial_parameters, coordinate[i], (template_list[i].shape[1],template_list[i].shape[0]),max_iter=100, learning_rate=1)
    return initial_parameters

def projective_Pyramid(img_template, img_search, coordinate, dimension, no_layers,max_iter=100) :   
    
    initial_parameters = np.array([0,0,0,0,0,0,0,0])
    for i in range(no_layers-1,-1,-1):
        initial_parameters = 2 * initial_parameters
        initial_parameters[2] = initial_parameters[2]/4
        initial_parameters[5] = initial_parameters[5]/4
        lk.template = template_list[i]
        lk.rows, lk.cols = lk.template.shape
        initial_parameters = projective_tracker(img_template, img_search[i], initial_parameters, coordinate[i], (template_list[i].shape[1],template_list[i].shape[0]),max_iter=100, learning_rate=1)
    return initial_parameters
