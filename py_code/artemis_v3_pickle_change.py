import cv2
import pandas as pd
import numpy as np
from scipy import stats
import glob
import sys
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import simpledialog
import pyautogui
import pickle

class show_prediction():

    def __init__(self, path_to_video, path_to_csv, path_to_pickle):
        """
        Initialize variables for script and dictionary with
        keys to behavior label. You can set up this
        dictionary however you like - pickle file with save
        keys to frame, while values will be displayed on video.
        :param path_to_video:
        :param path_to_csv:
        :param path_to_pickle:
        """
        self.screen_width = pyautogui.size()[0]
        self.screen_height = pyautogui.size()[1]
        self.inst_loc = [int(self.screen_width / 2), 0]
        self.path_to_video = path_to_video
        self.path_to_csv = path_to_csv
        self.path_to_pickle = path_to_pickle
        os.chdir(self.path_to_csv)
        self.font = cv2.FONT_HERSHEY_COMPLEX
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
            9: "none"
        }
        # empty dataframe to store annotations for session
        self.annot_data = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
        self.annotating = True

    def show_intro(self):
        """
        Displays instructions.
        """
        ready = False
        ndarray = np.full((640, 900, 3), 0, dtype=np.uint8)
        title_image2 = cv2.putText(ndarray, "Instructions:",
                                   (20, 20), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image2 = cv2.putText(ndarray, "Press y if correct",
                                   (20, 70), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image3 = cv2.putText(ndarray, "Press correct number if incorrect",
                                   (20, 120), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "Press spacebar to start video and to replay",
                                   (20, 170), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "Press left arrow to rewind and right arrow to fast-forward",
                                   (20, 220), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "If you need to redo annotation, press left key, then correct number",
                                   (20, 270), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "___________________________________________________________________",
                                   (20, 370), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "Press Esc to quit",
                                   (20, 300), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "If multiple behaviors in loop, do not annotate!",
                                   (20, 500), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "If there is CSV file MAKE SURE TO PICK THE SAME CSV AND VIDEO FILE!",
                                   (20, 550), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow("Directions")
        cv2.moveWindow("Directions", self.inst_loc[0], self.inst_loc[1])
        cv2.imshow("Directions", ndarray)
        while ready == False:
            if cv2.waitKey(0) == ord(' '):
                ready = True
        cv2.destroyAllWindows()

    def load_video_organize_dir(self):
        """
        Double checks folders and creates if none.
        Checks if directory has videos.
        If no CSV file is picked, still loads video and annotates.
        Double checks folders and creates if none.
        :input:
        CSV file (corresponding to video file)
        Video file (corresponding to csv file)
        pickle file (directory to save annotations)
        :return:
        If CSV file picked returns loaded dataframe with annotations
        Camera object
        """
        try:
            os.mkdir(self.path_to_video + '/videos_done')
        except:
            pass
        try:
            os.mkdir(self.path_to_video + '/videos_not_done')
        except:
            if len(os.listdir(self.path_to_video + '/videos_not_done')) == 0:
                ndarray = np.full((640, 900, 3), 0, dtype=np.uint8)
                title_image4 = cv2.putText(ndarray, "Folder empty: transfer videos to folder. Press ESC.",
                                           (20, 400), self.font,
                                           0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Transfer Files", ndarray)
                if cv2.waitKey(0) == ord('\x1b'):
                    cv2.destroyAllWindows()
                    sys.exit()
            pass
        try:
            os.mkdir(self.path_to_csv + '/csv_not_done')
        except:
            if len(os.listdir(self.path_to_csv + '/csv_not_done')) == 0:
                ndarray = np.full((640, 900, 3), 0, dtype=np.uint8)
                title_image4 = cv2.putText(ndarray, "Folder empty: transfer csv to folder. Press ESC.",
                                           (20, 400), self.font,
                                           0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Transfer Files", ndarray)
                if cv2.waitKey(0) == ord('\x1b'):
                    cv2.destroyAllWindows()
                    sys.exit()
            pass
        try:
            os.mkdir(self.path_to_csv + '/csv_done')
        except:
            pass
        try:
            os.mkdir(self.path_to_pickle + '/pickle_files/train')
            os.mkdir(self.path_to_pickle + '/pickle_files/test')
        except:
            pass
        root = tk.Tk()
        root.withdraw()
        self.test_or_train = simpledialog.askstring(title="Test",
                                                 prompt="Add annotations to test or train dataset:")
        if self.test_or_train == 'test':
            self.pickle_path = self.path_to_pickle + "/pickle_files"
            pass
        else:
            self.boot_round = simpledialog.askstring(title="Test",
                                                 prompt="What bootstrap round is this?:")
            self.pickle_path = self.path_to_pickle + "/pickle_files"

        self.csv_file = askopenfilename(initialdir="/home/jordan/Desktop/nihgpppipe/Annot/csv_not_done",
                                        title="Select CSV file")  # show an "Open" dialog box and return the path to the selected file
        self.video_file = askopenfilename(initialdir="/home/jordan/Desktop/nihgpppipe/Annot/videos_not_done",
                                          title="Select VIDEO file")
        if self.csv_file:
            self.predictions = pd.read_csv(self.csv_file, names=['frame', 'pred'])
        self.cap = cv2.VideoCapture(self.video_file)

    def determine_last_frame(self):
        """
        Determines last frame. Basically a save mechanism so you don't have to start over.
        :return:
        start_frame to pass annotation gui.
        """
        self.frames_analyzed = []
        self.exp_frames_analyzed_list = []
        self.start_frame = []
        self.prediction_state = False
        for file in glob.glob(self.pickle_path + "/train"+ self.video_file[self.video_file.rfind('/'):-4] + '*'):
            self.annot_pickle = pd.read_pickle(file)
            self.annot_pickle.sort_values(by='frame',inplace=True)
            self.exp_frames_analyzed_list.append(self.annot_pickle)
            self.frames_analyzed.append(len(self.annot_pickle.index))
        for file in glob.glob(self.pickle_path + "/test"+ self.video_file[self.video_file.rfind('/'):-4] + '*'):
            self.annot_pickle = pd.read_pickle(file)
            self.annot_pickle.sort_values(by='frame',inplace=True)
            self.exp_frames_analyzed_list.append(self.annot_pickle)
            self.frames_analyzed.append(len(self.annot_pickle.index))
        try:
            self.exp_frames_analyzed = pd.concat(self.exp_frames_analyzed_list, ignore_index=True)
            self.exp_frames_analyzed['frame'] = self.exp_frames_analyzed['frame'].astype('int32')
            common = self.predictions.merge(self.exp_frames_analyzed, on=['frame'])
            self.non_analyzed_frames = self.predictions[(~self.predictions.frame.isin(common.frame))]
            self.non_analyzed_frames.drop_duplicates(subset=['frame'])
            self.non_analyzed_frames.sort_values(by='frame', inplace=True)
            self.pred_dict = pd.Series(self.non_analyzed_frames.pred.values,
                                       index=self.non_analyzed_frames.frame).to_dict()
            self.prediction_state = True
        except:
            pass
        try:
            if self.prediction_state == False:
                self.pred_dict = pd.Series(self.predictions.pred.values, index=self.predictions.frame).to_dict()
                self.prediction_state = True
        except:
            pass

        if self.test_or_train == 'test':
            try:
                self.annot_pickle = pd.read_pickle(
                    self.pickle_path + "/test" + self.video_file[self.video_file.rfind('/'):-4] + '_test.p')
                self.annot_pickle.sort_values(by='frame',inplace=True)
                self.annot_pickle.drop_duplicates(subset=['frame'])
                pickl_pres = True
            except:
                self.annot_pickle = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
                pickl_pres = False
                pass
            if pickl_pres == True:
                try:
                    self.start = (self.non_analyzed_frames['frame'].iloc[0])
                except:
                    a = self.annot_pickle['frame'].iloc[0]
                    for row, index in self.annot_pickle.iterrows():
                        if index['frame'] - a > 1:
                            self.start = a + 1
                            break
                        a = index['frame']
                        self.start = self.annot_pickle['frame'].iloc[-1] + 1
            if pickl_pres == False:
                self.start = 80
        else:
            try:
                self.annot_pickle = pd.read_pickle(
                    self.pickle_path + "/train" + self.video_file[self.video_file.rfind('/'):-4] + '_boot{}.p'.format(self.boot_round))
                self.annot_pickle.sort_values(by='frame', inplace=True)
                self.annot_pickle.drop_duplicates(subset=['frame'])
                pickl_pres = True
            except:
                self.annot_pickle = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
                pickl_pres = False
                pass
            if pickl_pres == True:
                try:
                    self.start = (self.non_analyzed_frames['frame'].iloc[0])
                except:
                    a = self.annot_pickle['frame'].iloc[0]
                    for row, index in self.annot_pickle.iterrows():
                        if index['frame'] - a > 1:
                            self.start = a + 1
                            break
                        a = index['frame']
            if pickl_pres == False:
                self.start = 80

        return self.start

    def loop_video(self, start_frame=80, interval=100):
        """
        Loops over video with gui. This is where you update or confirm annotations.
        :param start_frame:
        :param interval:
        :return:
        Appends annotation to pandas dataframe
        """
        self.start_frame = (start_frame)
        self.end_frame = self.start_frame + (interval)
        self.made_pred = True
        if (self.cap.isOpened() == False):
            print("Error opening video stream or file")
        self.cap.set(1, self.start_frame)
        while (self.cap.isOpened()):
            if int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - self.start_frame < interval:
                ret, frame = self.cap.read()
                self.det_pred = self.determine_prediction(self.start_frame, self.end_frame - 1)
                if self.det_pred == 10:
                    self.loop_video((self.start_frame + interval))
                frame_pred = cv2.putText(frame, "Start Frame: {}".format(self.start_frame) + "   Pred: " +
                                         self.BEHAVIOR_LABELS[int(self.det_pred)], (100, 25),
                                         cv2.FONT_HERSHEY_DUPLEX, 0.75,
                                         (60, 76, 231), 1, cv2.LINE_AA)
                cv2.namedWindow('image')
                cv2.moveWindow('image', 0, 0)
                cv2.imshow('image', frame)
                cv2.waitKey(int(100 / 24))
            else:
                if self.annotating == True:
                    k = cv2.waitKey(100)
                    ndarray = np.full((640, 900, 3), 0, dtype=np.uint8)
                    title_image2 = cv2.putText(ndarray, "What was the behavior?",
                                               (20, 150), self.font,
                                               0.7, (255, 255, 255), 1, cv2.LINE_AA)
                    title_image2 = cv2.putText(ndarray, "0: drink, 1: eat, 2: groom, 3: hang, 4: sniff",
                                               (20, 200), self.font,
                                               0.7, (255, 255, 255), 1, cv2.LINE_AA)
                    title_image2 = cv2.putText(ndarray, "5: rear, 6: rest, 7: walk, 8: eathand, 9: no-pred",
                                               (20, 250), self.font,
                                               0.7, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.namedWindow("Ins1")
                    cv2.moveWindow("Ins1", self.inst_loc[0], self.inst_loc[1])
                    cv2.imshow("Ins1", ndarray)
                    k = cv2.waitKey(0)
                    if k & 0xFF == ord('0'):
                        self.det_pred = int(chr(k))
                        self.update_annotations()
                        self.loop_video((self.start_frame + interval))
                    elif k & 0xFF == ord('\x1b'):
                        self.annotating = False
                        cv2.destroyAllWindows()
                        self.det_pred = None
                        self.save_annotations_as_pickle()
                    elif k & 0xFF == ord(' '):
                        self.loop_video(self.start_frame)
                    elif k & 0xFF == ord('Q'):
                        self.loop_video(self.start_frame - interval)
                    elif k & 0xFF == ord('S'):
                        self.loop_video((self.start_frame + interval))
                    elif k & 0xFF == ord('1') or k & 0xFF == ord('2') or k & 0xFF == ord('3') or k & 0xFF == ord(
                            '4') or k & 0xFF == ord('5') or k & 0xFF == ord('6') or k & 0xFF == ord(
                            '7') or k & 0xFF == ord('8') or k & 0xFF == ord('9'):
                        self.det_pred = int(chr(k))
                        self.update_annotations()
                        self.loop_video((self.start_frame + interval))
                    elif k & 0xFF == ord('y'):
                        self.update_annotations()
                        self.loop_video((self.start_frame + interval))

    def determine_prediction(self, start_frame, stop_frame):
        """
        Determines prediction over specific interval of video.
        :param start_frame:
        :param stop_frame:
        :return:
        Mode of predictions over specific interval. This is displayed on video
        """
        preds = []
        for pred in np.arange(start_frame, stop_frame + 1):
            try:
                preds.append(self.pred_dict[pred])
            except:
                continue
        if not self.csv_file:
            if start_frame not in self.annot_pickle['frame'].values:
                return 9
            else:
                return 10
        most_pred = stats.mode(preds, axis=None)
        if not most_pred[0]:
            return 10
        preds = []
        return int(most_pred[0])

    def update_annotations(self):
        """
        Called to update annotation to pandas dataframe.
        This loops over frame interval and creates a new row for each frame.
        :return:
        pandas data frame with columns [frame, pred]
        """
        for frame in np.arange(self.start_frame,self.end_frame):
            self.annot_data = self.annot_data.append(
                {'frame': frame, 'pred': self.BEHAVIOR_LABELS[int(self.det_pred)]}, ignore_index=True)

    def save_annotations_as_pickle(self):
        """
        Converts annotations to pickle file.
        :return:
        pickle file with columns [frame, pred]
        """
        self.annot_pickle_final = pd.concat([self.annot_pickle, self.annot_data])
        self.annot_pickle_final.sort_values(by='frame',inplace=True)
        if self.test_or_train == 'test':
            self.annot_pickle_final.to_pickle(self.pickle_path + "/test"+self.video_file[self.video_file.rfind('/'):-4] + '_test.p')
            a = pd.read_pickle(self.pickle_path + "/test"+self.video_file[self.video_file.rfind('/'):-4] + '_test.p')
            print(a)
        else:
            self.annot_pickle_final.to_pickle(self.pickle_path +"/train"+ self.video_file[self.video_file.rfind('/'):-4] + '_boot{}.p'.format(self.boot_round))
            b = pd.read_pickle(self.pickle_path +"/train"+ self.video_file[self.video_file.rfind('/'):-4] + '_boot{}.p'.format(self.boot_round))
            print(b)
        self.done_with_video()

    def done_with_video(self):
        """
        Asks if you are done with video and diplays percentage of video analyzed.
        Either moves video and csv file to done directory, or keeps in not_done directory.
        """
        frame_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            self.frames_analyzed.append(len(self.annot_data.index))
        except:
            pass
        analyzed_frames = np.sum(self.frames_analyzed)
        ndarray2 = np.full((640, 900, 3), 0, dtype=np.uint8)

        title_image3 = cv2.putText(ndarray2, "You have analyzed {} percent of video".format(
            (analyzed_frames / frame_total) * 100),
                                   (20, 150), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image3 = cv2.putText(ndarray2, "Press m to move data to done folder",
                                   (20, 200), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image3 = cv2.putText(ndarray2, "Press s to keep video for more analysis",
                                   (20, 250), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow("Ins3")
        cv2.moveWindow("Ins3", self.inst_loc[0], self.inst_loc[1])
        cv2.imshow("Ins3", ndarray2)
        while True:
            j = cv2.waitKey(0)
            if j == ord('m'):
                os.rename(self.video_file,
                          self.path_to_video + '/videos_done' + self.video_file[self.video_file.rfind('/'):])
                os.rename(self.csv_file, self.path_to_csv + '/csv_done' + self.csv_file[self.csv_file.rfind('/'):])
                sys.exit()
            elif j == ord('s'):
                sys.exit()
if __name__ == "__main__":
    a = show_prediction('/home/jordan/Desktop/nihgpppipe/Annot', '/home/jordan/Desktop/nihgpppipe/Annot',
                        '/home/jordan/Desktop/nihgpppipe/Annot')
    a.show_intro()
    a.load_video_organize_dir()
    last_frame = a.determine_last_frame()
    a.loop_video(last_frame)

#fixed with sort, need to figure out why train always starts at 7679 (think because it is not deleting first couple frames) - drop duplicates and add frame 7678 to annot_pickle
#add messages for esceptions
#cut down onf code

