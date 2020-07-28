import cv2
import tkinter as tk
from tkinter import *
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd


class display:

    def __init__(self):
        self.csv = None
        self.max_frame = None
        self.max_labelled = None
        self.BEHAVIOR_LABELS = {
            0: "drink",
            2: "groom",
            1: "eat",
            3: "hang",
            4: "sniff",
            5: "rear",
            6: "rest",
            7: "walk",
            8: "eathand",
            9: "none",
            10: "N/A"
        }
        self.font = cv2.FONT_HERSHEY_COMPLEX

    def intro(self, img=np.full((640, 900, 3), 0, dtype=np.uint8)):

        ready = False

        directions = cv2.putText(img, "Instructions:",
                                 (20, 20), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "Press y if correct",
                                 (20, 70), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "Press correct number if incorrect",
                                 (20, 120), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "Press spacebar to start video and to replay",
                                 (20, 170), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "Press left arrow to rewind and right arrow to fast-forward",
                                 (20, 220), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "If you need to redo annotation, press left key, then correct number",
                                 (20, 270), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "___________________________________________________________________",
                                 (20, 370), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "Press Esc to quit",
                                 (20, 300), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "If multiple behaviors in loop, do not annotate!",
                                 (20, 400), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "If there is CSV file MAKE SURE TO PICK THE SAME CSV AND VIDEO FILE!",
                                 (20, 450), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img, "ONCE YOU SAVE PICKLE FILE, YOU CAN'T CHANGE ANNOTATIONS.",
                                 (20, 500), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        directions = cv2.putText(img,
                                 "MAKE SURE THEY ARE CORRECT BEFORE SAVE!",
                                 (20, 525), self.font,
                                 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow("Directions")
        cv2.imshow("Directions", img)
        while ready == False:
            if cv2.waitKey(0) == ord(' '):
                ready = True
        cv2.destroyAllWindows()

    def pick_file(self, main_path, video_files=None, rsync_path=None):
        if rsync_path is not None:
            self.root = tk.Tk()
            for video in video_files:
                button = Button(self.root, text=video,
                                command=lambda x=video: self.pick_video(x, main_path, rsync_path))
                button.pack()
            self.root.mainloop()
            root = tk.Tk()
            root.withdraw()
            test_or_train = simpledialog.askstring(title="Test",
                                                   prompt="Add annotations to test or train dataset:")
            return self.video_file, self.video_rsync, test_or_train

    def pick_video(self, name, main_path, rsync_path):
        video_file = name
        self.video_file = (main_path + "/videos_not_done/{}".format(video_file))
        self.video_rsync = (rsync_path + "/videos_not_done/{}".format(video_file))
        self.root.destroy()

    def choose_local_file(self, main_path):
        root = tk.Tk()
        root.withdraw()
        test_or_train = simpledialog.askstring(title="Test",
                                               prompt="Add annotations to test or train dataset:")
        video_file = askopenfilename(initialdir=main_path,
                                     title="Select VIDEO file")
        return video_file, test_or_train

    def setup_video_properties(self, video, csv_df):
        """
        Sets values to attribute variables.
        :param video: Video array.
        :param csv_path: Path to csv file containing predictions.
        """
        # TODO: Make sure frame types are set to int-64.
        self.csv = csv_df
        self.csv.columns = ['frame', 'pred']
        self.max_frame = len(video)
        self.max_labelled = len(self.csv)

    def video_loop(self, video, start, csv_path, interval=100, fps=30):
        """
        Loops over interval-many frames in the video
        :param start: Start of interval.
        :param csv_path: Path to csv file containing predictions.
        :param fps: Frames per second to loop at.
        :param interval: Length of interval.
        :param video: Array of images using pims.
        """
        print(f'DISPLAY: Playing video at start frame {start}')
        end_frame = min(start + interval, self.max_frame)
        milliseconds_per_frame = round(1000 / fps)  # ms/sec * sec/frame
        # Black image in case no video frames are able to be shown
        img = np.full((640, 900, 3), 0, dtype=np.uint8)
        for i in range(start, end_frame):
            img = video[i]
            prediction_number = self.csv.iloc[i]['pred']
            # Prediction is N/A by default. If prediction exists (index is within number of labelled frames),
            #  prediction string is changed.
            if i < self.max_labelled:
                prediction = self.BEHAVIOR_LABELS[prediction_number]
            else:
                prediction = "N/A"

            frame_pred = cv2.putText(img, "Current: Frame " + str(i) + "   Pred: " +
                                     prediction, (5, 25),
                                     cv2.FONT_HERSHEY_DUPLEX, 0.75,
                                     (60, 76, 231), 1, cv2.LINE_AA)
            cv2.namedWindow('image')
            cv2.imshow('image', img)
            cv2.waitKey(milliseconds_per_frame)

        frame_pred = cv2.putText(img, "Loop Done.", (5, 50),
                                 cv2.FONT_HERSHEY_DUPLEX, 0.75,
                                 (60, 76, 231), 1, cv2.LINE_AA)

        nd_array = np.full((640, 900, 3), 0, dtype=np.uint8)
        title_image2 = cv2.putText(nd_array, "What was the behavior?",
                                   (20, 150), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image2 = cv2.putText(nd_array, "0: drink, 1: eat, 2: groom, 3: hang, 4: sniff",
                                   (20, 200), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image2 = cv2.putText(nd_array, "5: rear, 6: rest, 7: walk, 8: eathand, 9: no-pred",
                                   (20, 250), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow("Ins1")
        cv2.imshow("Ins1", nd_array)

    def done_with_video(self, total_frames, pickle_df, percent=None):

        total_annotated_overall = len(pickle_df)
        percent_analyzed = (total_annotated_overall / total_frames) * 100

        if percent is not None:
            percent_analyzed = percent

        ndarray2 = np.full((640, 900, 3), 0, dtype=np.uint8)

        title_image3 = cv2.putText(ndarray2, "You have analyzed {} percent of video".format(percent_analyzed),
                                   (20, 150), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image3 = cv2.putText(ndarray2, "Press s to keep video for more analysis",
                                   (20, 250), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image3 = cv2.putText(ndarray2, "Press m to move video to done folder",
                                   (20, 300), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow("Ins3")
        cv2.imshow("Ins3", ndarray2)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return k
