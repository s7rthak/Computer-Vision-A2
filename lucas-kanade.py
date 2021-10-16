import numpy as np
import cv2

ST = 1
NU = 2
TU = 3

BLURCAR = 585

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

class LucasKanade():
    def __init__(self, template, bbox, update_type):
        self.template = template
        self.bbox = bbox
        self.update_type = update_type
        self.p_prev = np.zeros(6)
        self.rows, self.cols = template.shape

    def get_best_match(self, frame):
        Iy, Ix = np.gradient(frame)
        p = self.p_prev

        cnt = 0
        while cnt < 100:
            cnt += 1
            WM = np.array([[1.0 + p[0], p[1], p[2]],
                       [p[3], 1.0 + p[4], p[5]]])

            warpImg = cv2.warpAffine(frame, WM, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)[self.bbox[1]:self.bbox[1]+self.bbox[3],self.bbox[0]:self.bbox[0]+self.bbox[2]]
            Ix_g = cv2.warpAffine(Ix, WM, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)
            Iy_g = cv2.warpAffine(Iy, WM, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)
            I_g = np.vstack((Ix_g.ravel(),Iy_g.ravel())).T

            err = self.template - warpImg
            errImg = err.reshape(-1, 1)

            delta = np.zeros((self.template.shape[0]*self.template.shape[1], 6))

            for i in range(self.template.shape[0]):
                for j in range(self.template.shape[1]):
                    I_indiv = np.array([I_g[i*self.template.shape[1]+j]]).reshape(1, 2)
                    jac_indiv = np.array([[j, 0, i, 0, 1, 0],
                                            [0, j, 0, i, 0, 1]])
                    delta[i*self.template.shape[1]+j] = I_indiv @ jac_indiv

            H = delta.T @ delta

            dp = np.linalg.pinv(H) @ (delta.T) @ errImg
            print(dp)

            p[0] += dp[0,0]
            p[1] += dp[1,0]
            p[2] += dp[2,0]
            p[3] += dp[3,0]
            p[4] += dp[4,0]
            p[5] += dp[5,0]

        self.p_prev = p

        W = np.array([[1.0 + p[0], p[1], p[2]],
                       [p[3], 1.0 + p[4], p[5]]])
        points = [(self.bbox[0],self.bbox[1]), (self.bbox[0]+self.bbox[2],self.bbox[1]), (self.bbox[0]+self.bbox[2],self.bbox[1]+self.bbox[3]), (self.bbox[0],self.bbox[1]+self.bbox[3])]
        print(points)
        print(W)
        for i in range(4):
            warpedPt = (round(W[0,0] * points[i][0] + W[0,1] * points[i][1] + W[0,2]), round(W[1,0] * points[i][0] + W[1,1] * points[i][1] + W[1,2]))
            points[i] = warpedPt
        
        return points

bboxes = []
with open('A2/BlurCar2/groundtruth_rect.txt') as gt:
    all_lines = gt.readlines()
    for line in all_lines:
        nums = line.split('\t')
        bboxes.append((int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])))

first_frame = cv2.imread('A2/BlurCar2/img/0001.jpg')
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
lk = LucasKanade(first_frame[bboxes[0][1]:bboxes[0][1]+bboxes[0][3],bboxes[0][0]:bboxes[0][0]+bboxes[0][2]], bboxes[0], ST)

for i in range(2, BLURCAR+1):
    print(i)
    frame = cv2.imread('A2/BlurCar2/img/' + str(i).zfill(4) + '.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_match = lk.get_best_match(frame)
    print(best_match)
    frame = cv2.line(frame, best_match[0], best_match[1], (255,0,0), 2)
    frame = cv2.line(frame, best_match[1], best_match[2], (255,0,0), 2)
    frame = cv2.line(frame, best_match[2], best_match[3], (255,0,0), 2)
    frame = cv2.line(frame, best_match[3], best_match[0], (255,0,0), 2)
    cv2.imwrite('A2/BlurCar2/output/' + str(i).zfill(4) + '.jpg', frame)