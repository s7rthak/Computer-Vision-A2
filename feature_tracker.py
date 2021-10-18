import cv2
import numpy as np

MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.45
BLURCAR = 585

class TemplateTracker():
    def __init__(self, template, bbox):
        self.template = template
        self.bbox = bbox
        self.orb = cv2.ORB_create(int(MAX_FEATURES/10))
        self.template_keypoints, self.template_desc = self.orb.detectAndCompute(template, None)
        print(self.template_keypoints)
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)

    def warped_bbox(self, h):
        points = [(self.bbox[0],self.bbox[1]), (self.bbox[0]+self.bbox[2],self.bbox[1]), (self.bbox[0]+self.bbox[2],self.bbox[1]+self.bbox[3]), (self.bbox[0],self.bbox[1]+self.bbox[3])]
        
        for i, p in enumerate(points):
            px = (h[0][0]*p[0] + h[0][1]*p[1] + h[0][2]) / ((h[2][0]*p[0] + h[2][1]*p[1] + h[2][2]))
            py = (h[1][0]*p[0] + h[1][1]*p[1] + h[1][2]) / ((h[2][0]*p[0] + h[2][1]*p[1] + h[2][2]))
            points[i] = (round(px),round(py))

        return points

    def get_best_template_match_bbox(self, frame):
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints, descriptors = orb.detectAndCompute(frame, None)

        matches = self.matcher.match(self.template_desc, descriptors, None)
        matches.sort(key=lambda x: x.distance, reverse=False)

        good_matches = matches[:int(len(matches) * GOOD_MATCH_PERCENT)]
        print(len(good_matches))

        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        shift = np.array([self.bbox[0],self.bbox[1]])
        for i, match in enumerate(good_matches):
            points1[i, :] = self.template_keypoints[match.queryIdx].pt + np.array([self.bbox[0],self.bbox[1]])
            points2[i, :] = keypoints[match.trainIdx].pt

        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # imMatches = cv2.drawMatches(self.template, self.template_keypoints, frame, keypoints, good_matches, None)
        # cv2.imshow("match", imMatches)
        # cv2.waitKey(2000)

        return self.warped_bbox(h)


bboxes = []
with open('A2/BlurCar2/groundtruth_rect.txt') as gt:
    all_lines = gt.readlines()
    for line in all_lines:
        nums = line.split('\t')
        bboxes.append((int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])))

first_frame = cv2.imread('A2/BlurCar2/img/0001.jpg')
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
tt = TemplateTracker(first_frame[bboxes[0][1]:bboxes[0][1]+bboxes[0][3],bboxes[0][0]:bboxes[0][0]+bboxes[0][2]], bboxes[0])

for i in range(2, BLURCAR+1):
    print(i)
    frame = cv2.imread('A2/BlurCar2/img/' + str(i).zfill(4) + '.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    best_match = tt.get_best_template_match_bbox(frame)
    print(best_match)
    print(bboxes[i-1])
    frame = cv2.line(frame, best_match[0], best_match[1], (255,0,0), 2)
    frame = cv2.line(frame, best_match[1], best_match[2], (255,0,0), 2)
    frame = cv2.line(frame, best_match[2], best_match[3], (255,0,0), 2)
    frame = cv2.line(frame, best_match[3], best_match[0], (255,0,0), 2)
    cv2.imwrite('A2/BlurCar2/output/' + str(i).zfill(4) + '.jpg', frame)