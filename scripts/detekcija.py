#!/usr/bin/env python
import time
import numpy as np
import cv2
import roslib
import rospy

from sensor_msgs.msg import CompressedImage


class DemageDetection:
    def __init__(self):
        self.subscriber = rospy.Subscriber("/bebop/image_raw/compressed",
            CompressedImage, self.img_cm_callback,  queue_size = 1)
        self.COLOR_NAMES = ["red"]
        self.COLOR_RANGES_HSV = {
        "red": [(0, 50, 10), (10, 255, 255)]
    }

    def img_cm_callback(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        self.image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.detect()
        final_frame = cv2.hconcat((self.image_np, self.imgCopy))
        size_pic = (1700, 650)
        final_frame = cv2.resize(final_frame, size_pic)
        cv2.imshow('frame', final_frame)
        cv2.waitKey(2)


    def detect(self):
        self.imgCopy = self.image_np.copy()
        circles = self.detectCirclesWithDp()
        if circles is not None:
            for circle in circles[0, :]:
                roi = self.getROI(self.imgCopy, circle[0], circle[1], circle[2])
                color = self.getDominantColor(roi)
                cv2.circle(self.imgCopy, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                # cv2.circle(self.imgCopy, (circle[0], circle[1]), 2, (0, 255, 0), 2)
                cv2.putText(self.imgCopy, "ostecenje", (int(circle[0] + 40), int(circle[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0))

    def detectCirclesWithDp(self):
        frame = cv2.cvtColor(self.image_np, cv2.COLOR_BGR2HSV)
        blurred = cv2.medianBlur(frame, 5)
        grayMask = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        return cv2.HoughCircles(grayMask, cv2.HOUGH_GRADIENT, 1, 40, param1=100, param2=40, minRadius=10, maxRadius=200)
    
    def getROI(self, frame, x, y, r):
        return frame[int(y-r/2):int(y+r/2), int(x-r/2):int(x+r/2)]

    def getMask(self, frame, color):
        blurredFrame = cv2.GaussianBlur(frame, (3, 3), 0)
        hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

        colorRange = self.COLOR_RANGES_HSV[color]
        lower = np.array(colorRange[0])
        upper = np.array(colorRange[1])

        colorMask = cv2.inRange(hsvFrame, lower, upper)
        colorMask = cv2.bitwise_and(blurredFrame, blurredFrame, mask=colorMask)

        return colorMask

    def getDominantColor(self, roi):
        roi = np.float32(roi)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        ret, label, center = cv2.kmeans(roi, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(roi.shape)

        pixelsPerColor = []
        for color in self.COLOR_NAMES:
            mask = self.getMask(res2, color)
            greyMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            count = cv2.countNonZero(greyMask)
            pixelsPerColor.append(count)

        return self.COLOR_NAMES[pixelsPerColor.index(max(pixelsPerColor))]
        
        
def main():
    rospy.init_node('demage_detection', anonymous=True)
    ic = DemageDetection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

    # def Detekcija(self):
    #     self.output = self.image_np.copy()
    #     imgGry = cv2.cvtColor(self.image_np, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.medianBlur(imgGry, 5)
    #     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30,
    #                             param1=100, param2=30, minRadius=1, maxRadius=200)
    #     detected_circles = np.uint16(np.around(circles))
    #     for (x, y ,r) in detected_circles[0, :]:
    #         cv2.circle(self.output, (x, y), r, (0, 255, 0), 2)
    #         # cv2.circle(self.self.output, (x, y), 2, (0, 255, 0), 2)
    #         cv2.putText(self.output, "Ostecenje", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))

    # def Detekcija1(self):

    #     self.original = self.image_np.copy()
    #     image = cv2.cvtColor(self.image_np, cv2.COLOR_BGR2HSV)
    #     # Red
    #     lower = np.array([0, 50, 10], dtype="uint8")
    #     upper = np.array([10, 255, 255], dtype="uint8")
    #     mask = cv2.inRange(image, lower, upper)

    #     # Find contours
    #     cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # Extract contours depending on OpenCV version
    #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    #     # Iterate through contours and filter by the number of vertices 
    #     for c in cnts:
    #         perimeter = cv2.arcLength(c, True)
    #         approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
    #         if len(approx) > 5 and len(approx) < 10:
    #             cv2.drawContours(self.original, [c], -1, (36, 255, 12), -1)