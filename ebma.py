import os
import cv2
import numpy as np

SSD = 1
NCC = 2
BOLT = 350

def iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    x_right = min(bb1[0]+bb1[2], bb2[0]+bb2[2])
    y_top = max(bb1[1],bb2[1])
    y_bottom = min(bb1[1]+bb1[3], bb2[1]+bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = bb1[2] * bb1[3]
    bb2_area = bb2[2] * bb2[3]

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou

class BlockMatch():
    def __init__(self, template, bbox, alpha=0.7):
        self.template = template
        self.alpha = alpha
        self.bbox = bbox
        self.h = self.bbox[3]
        self.w = self.bbox[2]

    def get_best_match(self, frame, search_over=10000, stride=1, measure=SSD):
        H, W = frame.shape[0], frame.shape[1]
        C = frame.shape[2] if len(frame.shape) == 3 else 1

        min_diff = float('inf')
        max_diff = float('-inf')
        current_best_match = None

        min_x_range = max(0, self.bbox[0]-search_over)
        max_x_range = min(W - (self.w-1), self.bbox[0]+search_over)
        min_y_range = max(0, self.bbox[1]-search_over)
        max_y_range = min(H - (self.h-1), self.bbox[1]+search_over)

        for i in range(min_y_range, max_y_range, stride):
            for j in range(min_x_range, max_x_range, stride):
                current_block = frame[i:i+self.h,j:j+self.w].astype(np.int64)

                if measure == SSD:
                    current_diff = (1.0/(self.h*self.w))*np.square(current_block-self.template)
                    consolidated_diff = np.sum(current_diff)

                    if consolidated_diff < min_diff:
                        min_diff = consolidated_diff
                        current_best_match = (j, i, self.w, self.h)

                if measure == NCC:
                    normalized_img = np.zeros((self.h, self.w, C))
                    if C == 1:
                        normalized_img = (current_block - np.mean(current_block))/np.std(current_block)
                    else:
                        for i in range(C):
                            normalized_img[:,:,i] = (current_block[:,:,i] - np.mean(current_block[:,:,i].flatten()))/np.std(current_block[:,:,i].flatten())
                    
                    consolidated_diff = np.sum((self.template.flatten()).dot(normalized_img.flatten().T))
                    if consolidated_diff > max_diff:
                        max_diff = consolidated_diff
                        current_best_match = (j, i, self.w, self.h)

        # Template Update (Moving-Average)
        best_match_block = frame[current_best_match[1]:current_best_match[1]+self.h,current_best_match[0]:current_best_match[0]+self.w]
        self.template = np.rint(self.alpha * best_match_block + (1-self.alpha) * self.template)
        self.template = self.template.astype(np.uint8)
        self.bbox = current_best_match

        return current_best_match

bboxes = []
with open('A2/Bolt/groundtruth_rect.txt') as gt:
    all_lines = gt.readlines()
    for line in all_lines:
        nums = line.split(',')
        bboxes.append((int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])))

first_frame = cv2.imread('A2/Bolt/img/0001.jpg')
ebma = BlockMatch(first_frame[bboxes[0][1]:bboxes[0][1]+bboxes[0][3],bboxes[0][0]:bboxes[0][0]+bboxes[0][2]], bboxes[0], alpha=0.0)

IOUs = []
for i in range(2, BOLT+1):
    print(i)
    frame = cv2.imread('A2/Bolt/img/' + str(i).zfill(4) + '.jpg')
    best_match = ebma.get_best_match(frame, 5)
    IOUs.append(iou(best_match, bboxes[i-1]))
    tracked_frame = cv2.rectangle(frame, (best_match[0], best_match[1]), (best_match[0]+best_match[2], best_match[1]+best_match[3]), (255, 0, 0), 2)
    cv2.imwrite('A2/Bolt/output/' + str(i).zfill(4) + '.jpg', tracked_frame)

print('mIOU = ' + str(np.mean(IOUs)))