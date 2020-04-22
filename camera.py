import threading
import binascii
import cv2
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64
import numpy as np
from PIL import Image
from gaze_tracking import GazeTracking
from scipy.spatial import distance

gaze = GazeTracking()
frameRate = 20.0

class Camera(object):
    def __init__(self, makeup_artist):
        self.to_process = []
        self.to_output = []
        self.makeup_artist = makeup_artist

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string. 
        input_str = self.to_process.pop(0)

        # convert it to a pil image
        input_img = base64_to_pil_image(input_str)

        ################## where the hard work is done ############
        # output_img is an PIL image
        output_img = self.makeup_artist.apply_makeup(input_img)
        opencvImage = cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR)
        cv2.resize(opencvImage, (400,560), interpolation=cv2.INTER_AREA)

        im = cv2.imread("newmask.png")

        # cv2.rectangle(image, (400, 300), (700, 500), (178, 190, 181), 5)

        frame = cv2.flip(opencvImage, 2)

        gaze.refresh(frame)

        frame, x, y = gaze.annotated_frame()
        text = ""

        # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()

        right_pupil = gaze.pupil_right_coords()
        print(right_pupil, left_pupil)

        points_cnt = (x, y)

        # print(points_cnt)
        if left_pupil and right_pupil != None:
            a = left_pupil
            b = right_pupil
            c = points_cnt
            # dist = [(a - c) ** 2 for a, c in zip(a, c)]
            # dist = math.sqrt(sum(dist))
            # print("new method",dist)

            dst_left = distance.euclidean(a, c)
            mm = 0.26458333
            dist_left_mm = (dst_left * mm) + 20


            # print(dist_left_mm)
            print("left:::", dist_left_mm)
            dst_right = distance.euclidean(b, c)
            # print(dst_right)
            # print(dst_left)

            dist_right_mm = (dst_right * mm) + 20
            total_pd = dist_right_mm + dist_left_mm
            print("total::", total_pd)
            print("right::", dist_right_mm)

            cv2.putText(frame, "Left PD:  " + str(dist_left_mm) + 'mm', (80, 135), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                        (0, 0, 255), 1)
            cv2.putText(frame, "Right PD: " + str(dist_right_mm) + 'mm', (85, 175), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                        (0, 0, 255), 1)
            # cv2.putText(frame, "Total PD: " + str(total_pd) + 'mm', (85, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9,
            #             (0, 0, 255), 1)
        # print(frame.shape[1::-1])
        im = cv2.resize(im, frame.shape[1::-1], interpolation=cv2.INTER_AREA)
        dst = cv2.addWeighted(frame, 0.5, im, 0.5, 0)

        # cv2.imshow("Frame",dst)
        # cv2.waitKey(1)
        ##Convert opencv output to pillow image
        img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        # output_str is a base64 string in ascii
        output_str = pil_image_to_base64(im_pil)

        # convert eh base64 string in ascii to base64 string in _bytes_
        self.to_output.append(binascii.a2b_base64(output_str))

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)
