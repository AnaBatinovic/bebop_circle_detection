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
        # final_frame = cv2.hconcat((self.image_np, self.img))
        size_pic = (1700, 1000)
        final_frame = cv2.resize(self.img, size_pic)
        cv2.imshow('frame', final_frame)
        cv2.waitKey(2)

    def detect(self):
        self.img = self.image_np.copy()
        # blurring the frame that's captured
        frame_gau_blur = cv2.GaussianBlur(self.img, (3, 3), 0)
        # converting BGR to HSV
        hsv = cv2.cvtColor(frame_gau_blur, cv2.COLOR_BGR2HSV)
        # the range of red color in HSV
        lower_red = np.array([0, 50, 10])
        higher_red = np.array([10, 255, 255])
        # getting the range of red color in frame
        red_range = cv2.inRange(hsv, lower_red, higher_red)
        res_red = cv2.bitwise_and(frame_gau_blur,frame_gau_blur, mask=red_range)
        red_s_gray = cv2.cvtColor(res_red, cv2.COLOR_BGR2GRAY)
        canny_edge = cv2.Canny(red_s_gray, 50, 240)
        # applying HoughCircles
        circles = cv2.HoughCircles(canny_edge, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=200, param2=20, minRadius=3, maxRadius=30)
        cir_cen = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            counter = 0;
            for (x,y,r) in circles:
                ++counter
                if (counter <= 5):
                    cv2.circle(self.img, (x,y), r, (36,255,12), 2)
                    cv2.putText(self.img, "Ostecenje", (x+20, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                else:
                    break

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

