import cv2
import tkinter as tk
from tkinter import *
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename

class display():

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_COMPLEX

    def intro(self, img):
        ready=False
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
                button = Button(self.root, text=video, command=lambda x=video: self.pick_video(x, main_path, rsync_path))
                button.pack()
            self.root.mainloop()
            root = tk.Tk()
            root.withdraw()
            test_or_train = simpledialog.askstring(title="Test",
                                                        prompt="Add annotations to test or train dataset:")
            return self.video_file, self.video_rsync, test_or_train

    def pick_video(self,name, main_path, rsync_path):
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

    #def video_loop(self, img, x_text, y_text, wait_key_time):


    #def done_with_video(self, img, x_text, y_text, wait_ket_time):



