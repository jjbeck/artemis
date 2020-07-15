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
import argparse
import yaml
import re
import pims
import subprocess
from tkinter import *


class show_prediction():

    def __init__(self, main_path, rsync_path):
        """
        Initialize variables for script and dictionary with
        keys to behavior label. You can set up this
        dictionary however you like - pickle file with save
        keys to frame, while values will be displayed on video.
        :param main_path:
       :
        """
        self.back = False
        self.forward = False
        self.random = False
        self.main_path = main_path
        self.rsync_path = rsync_path
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
        self.pick_keys = {
            48:0,
            49:1,
            50:2,
            51:3,
            52:4,
            53:5,
            54:6,
            55:7,
            56:8,
            57:9,
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
                                   (20, 400), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "If there is CSV file MAKE SURE TO PICK THE SAME CSV AND VIDEO FILE!",
                                   (20, 450), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray, "ONCE YOU SAVE PICKLE FILE, YOU CAN'T CHANGE ANNOTATIONS.",
                                   (20, 500), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        title_image4 = cv2.putText(ndarray,
                                   "MAKE SURE THEY ARE CORRECT BEFORE SAVE!",
                                   (20, 525), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow("Directions")
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
        ls = subprocess.check_output("ssh jbecke11@serrep6.clps.brown.edu ls {}".format(self.main_path) + "/videos_not_done/", shell=True)
        video_files = ls.decode('utf-8').split('\n')

        self.root = tk.Tk()

        for video in video_files:
            button = Button(self.root,text=video, command=lambda x=video: self.pick_video(x))
            button.pack()

        self.root.mainloop()
        root = tk.Tk()
        root.withdraw()
        self.test_or_train = simpledialog.askstring(title="Test",
                                                 prompt="Add annotations to test or train dataset:")

        subprocess.run("rsync jbecke11@serrep6.clps.brown.edu:{} {}".format(self.main_path + "/config.yaml", self.rsync_path + "/config_yaml"), shell=True)
        with open(self.rsync_path + "/config.yaml") as file:
            config_param = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
        self.boot_round = config_param["Boot Round"]
        self.new_test_set_beh = config_param["Number of behaviors for Test Set"]["new"]
        self.old_test_set_beh = config_param["Number of behaviors for Test Set"]["old"]

        if self.test_or_train == "test":
            self.pickle_path = self.main_path + "/pickle_files/test" + self.video_file[self.video_file.rfind('/'):self.video_file.rfind(".")] + "_test.p"
            self.pickle_rsync = self.rsync_path + "/pickle_files/test" + self.video_file[self.video_file.rfind('/'):self.video_file.rfind(".")] + "_test.p"
        else:
            self.pickle_path = self.main_path + "/pickle_files/train" + self.video_file[self.video_file.rfind('/'):self.video_file.rfind(".")] + "_boot{}.p".format(self.boot_round)
            self.pickle_rsync = self.rsync_path + "/pickle_files/train" + self.video_file[self.video_file.rfind('/'):self.video_file.rfind(".")] + "_boot{}.p".format(self.boot_round)

        #self.video_file = askopenfilename(initialdir=self.main_path + "/videos_not_done",
                                          #title="Select VIDEO file")

        self.csv_file = self.main_path + "/csv_not_done" + self.video_file[self.video_file.rfind('/'):self.video_file.rfind(".")] + ".csv"
        self.csv_rsync = self.rsync_path + "/csv_not_done" + self.video_file[self.video_file.rfind('/'):self.video_file.rfind(".")] + ".csv"

        subprocess.run("rsync jbecke11@serrep6.clps.brown.edu:{} {}".format(self.csv_file, self.csv_rsync), shell=True)
        subprocess.run("rsync --progress jbecke11@serrep6.clps.brown.edu:{} {}".format(self.video_file, self.video_rsync), shell=True)
        #subprocess.run("rsync jbecke11@serrep6.clps.brown.edu:{} {}".format(self.video_file, self.video_rsync), shell=True)


        self.predictions = pd.read_csv(self.csv_rsync, names=['frame', 'pred'])

        try:
            self.pred_dict = pd.Series(self.predictions.pred.values,index=self.predictions.frame).to_dict()
            self.prediction_state = True
        except:
            pass

        self.cap = pims.PyAVReaderTimed(self.video_rsync)

        self.video_length = len(self.cap)

        frame_arr = np.arange(80,(self.video_length+1))
        self.total_frames = pd.DataFrame(data=frame_arr, columns=['frame'])
        self.video_file = self.video_file.replace(self.main_path,self.rsync_path)
        print(self.video_file)
        self.frame_start = subprocess.check_output('ssh jbecke11@serrep6.clps.brown.edu ". /home/jbecke11/andrew_holmes_pipe/bin/activate && python3 /media/data_cifs_lrs/projects/prj_nih/prj_andrew_holmes/artemis/py_code/calculate_frame_start.py -mp /media/data_cifs_lrs/projects/prj_nih/prj_andrew_holmes/ -vf {} -tt {} -br {}"'.format(self.video_file,self.test_or_train,self.boot_round), shell=True)
        self.frame_start = self.frame_start.decode('utf-8')
        return self.frame_start



    def pick_video(self,name):
        video_file = name
        self.video_file = (self.main_path + "/videos_not_done/{}".format(video_file))
        self.video_rsync = self.rsync_path + "/videos_not_done/{}".format(video_file)
        self.root.destroy()


    def determine_last_frame(self):


        """
        Determines last frame. Basically a save mechanism so you don't have to start over.
        :return:
        start_frame to pass annotation gui.
        
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
            self.non_analyzed_frames = pd.concat([self.total_frames,self.exp_frames_analyzed,self.exp_frames_analyzed],sort=True).drop_duplicates(subset=['frame'],keep=False)

        except:
            self.non_analyzed_frames = self.total_frames
            pass
        try:
            self.pred_dict = pd.Series(self.predictions.pred.values,index=self.predictions.frame).to_dict()
            self.prediction_state = True
        except:
            pass
        """

        if self.test_or_train == 'test':
            try:
                self.annot_pickle = pd.read_pickle(
                    self.pickle_rsync)
                self.annot_pickle.sort_values(by='frame',inplace=True)
                self.annot_pickle.drop_duplicates(subset=['frame'])
                pickl_pres = True
            except:
                self.annot_pickle = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
                pickl_pres = False
                pass
        else:
            try:
                self.annot_pickle = pd.read_pickle(
                    self.pickle_rsync)
                print("sync")
                self.annot_pickle.sort_values(by='frame', inplace=True)
                self.annot_pickle.drop_duplicates(subset=['frame'])
                pickl_pres = True
            except:
                self.annot_pickle = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
                pickl_pres = False
                pass

        print(f"Your current pickle file has {len(self.annot_pickle)} frames annotated")



    def choose_random_frame(self):
        a = self.non_analyzed_frames.sample()
        a = a["frame"]
        a = int(a)
        self.loop_video(a,self.interval,self.playback_speed)

    def loop_video(self, start_frame=80, interval=100, playback_speed = 1):
        """
        Loops over video with gui. This is where you update or confirm annotations.
        :param start_frame:
        :param interval:
        :return:
        Appends annotation to pandas dataframe
        """

        print(f"You have analyzed {len(self.annot_data)} frames in this session\r",end="")
        self.interval = interval
        self.start_frame = (start_frame)
        self.end_frame = self.start_frame + (self.interval)
        self.made_pred = True
        self.playback_speed = playback_speed
        while self.annotating == True:
            for i in np.arange(0,self.interval):
                img = self.cap[self.start_frame + i]
                self.det_pred = self.determine_prediction(self.start_frame, self.end_frame - 1)
                if self.det_pred == 10:
                    self.loop_video((self.start_frame + self.interval), self.interval)
                frame_pred = cv2.putText(img, "Current: Frame " + str(self.start_frame + i) +  "   Pred: " +
                                         self.BEHAVIOR_LABELS[int(self.det_pred)], (5, 25),
                                         cv2.FONT_HERSHEY_DUPLEX, 0.75,
                                         (60, 76, 231), 1, cv2.LINE_AA)
                cv2.namedWindow('image')
                cv2.imshow('image', img)
                cv2.waitKey(int((1 / (self.interval * self.playback_speed))*1000))
            else:
                if self.annotating == True:
                    frame_pred = cv2.putText(img, "Loop Done.", (5, 50),
                                             cv2.FONT_HERSHEY_DUPLEX, 0.75,
                                             (60, 76, 231), 1, cv2.LINE_AA)
                    cv2.imshow('image', img)
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
                    cv2.imshow("Ins1", ndarray)
                    k = cv2.waitKey(0)
                    if k & 0xFF == ord('0'):
                        self.det_pred = int(chr(k))
                        self.update_annotations()
                        self.random = False
                        self.forward = False
                        self.back = False
                        self.loop_video((self.start_frame + self.interval), self.interval, self.playback_speed)
                    elif k & 0xFF == ord('\x1b'):
                        self.annotating = False
                        cv2.destroyAllWindows()
                        self.det_pred = None
                        self.save_annotations_as_pickle()
                    elif k & 0xFF == ord(' '):
                        self.random = False
                        self.forward = False
                        self.back = False
                        self.loop_video(self.start_frame, self.interval, self.playback_speed)
                    elif k & 0xFF == ord('Q'):
                        self.back = True
                        self.loop_video(self.start_frame - self.interval, self.interval, self.playback_speed)
                    elif k & 0xFF == ord('S'):
                        self.forward = True
                        self.loop_video((self.start_frame + self.interval), self.interval, self.playback_speed)
                    elif k & 0xFF == ord('1') or k & 0xFF == ord('2') or k & 0xFF == ord('3') or k & 0xFF == ord(
                            '4') or k & 0xFF == ord('5') or k & 0xFF == ord('6') or k & 0xFF == ord(
                            '7') or k & 0xFF == ord('8') or k & 0xFF == ord('9'):
                        self.det_pred = int(chr(k))
                        self.update_annotations()
                        self.forward = False
                        self.back = False
                        self.random = False
                        self.loop_video((self.start_frame + self.interval), self.interval, self.playback_speed)
                    elif k & 0xFF == ord('y'):
                        self.update_annotations()
                        self.forward = False
                        self.back = False
                        self.random = False
                        self.loop_video((self.start_frame + self.interval), self.interval, self.playback_speed)
                    elif k & 0xFF == ord('r'):
                        self.random = True
                        self.choose_random_frame()
                    elif k & 0xFF == ord('s'):
                        self.forward = True
                        self.loop_video((self.start_frame + 1000), self.interval, self.playback_speed)
                    elif k & 0xFF == ord('p'):
                        print("picking")
                        j = cv2.waitKey(0)
                        beh = self.pick_keys[j]
                        con_data = pd.concat([self.annot_pickle, self.annot_data], sort=True)
                        non_analyzed_frames = pd.concat([self.predictions, con_data, con_data]).drop_duplicates(
                            subset=['frame'], keep='first')
                        beh_exist = non_analyzed_frames.where(non_analyzed_frames['pred'] == beh)
                        beh_exist = beh_exist.dropna()
                        try:
                            a = beh_exist.sample()
                            a = a["frame"]
                            a = int(a)
                            self.loop_video(a, self.interval, self.playback_speed)
                        except:
                            print("No {} left.".format(self.BEHAVIOR_LABELS[beh]))
                            self.loop_video(self.start_frame, self.interval, self.playback_speed)

        self.annotating = False
        cv2.destroyAllWindows()
        self.det_pred = None
        self.save_annotations_as_pickle()

    def choose_frame(self):
        beh = input("Enter behavior you would like to find: ")
        beh_to_num = { "drink": 0,
                        "eat": 1,
                        "groom": 2,
                        "hang": 3,
                        "sniff": 4,
                        "rear": 5,
                        "rest": 6,
                        "walk": 7,
                        "eathand":8,
                        "none": 9}

        con_data = pd.concat([self.annot_pickle,self.annot_data], sort=True)
        non_analyzed_frames = pd.concat([self.predictions,con_data,con_data]).drop_duplicates(subset=['frame'],keep=False)
        beh_exist = non_analyzed_frames.where(non_analyzed_frames['pred'] == beh_to_num[beh])
        beh_exist = beh_exist.dropna()
        print(beh_exist)
        a = beh_exist.sample()
        a = a["frame"]
        a = int(a)

        self.loop_video(a, self.interval, self.playback_speed)


    def determine_prediction(self, start_frame, stop_frame):
        """
        Determines prediction over specific interval of video.
        :param start_frame:
        :param stop_frame:
        :return:
        Mode of predictions over specific interval. This is displayed on video
        """
        #   last thing i need to MAKE IT SO PREDICTIONS CHECKS FOR FRAME STILL IN ANALYZED FRAMES AND NOT IN PREDICTIONS
        preds = []

        if not self.csv_file:
            if start_frame not in self.annot_pickle['frame'].values and start_frame not in self.annot_data['frame'].values:
                return 9
            elif self.back == True:
                a = self.annot_data.loc[self.annot_data['frame'] == start_frame]
                return list(self.BEHAVIOR_LABELS.keys())[
                    list(self.BEHAVIOR_LABELS.values()).index(a['pred'].to_string(index=False).strip())]
            elif self.forward == True:
                return 9
            else:
                return 10

        if self.random == True:
            if start_frame in self.annot_pickle['frame'].values and start_frame in self.annot_data['frame'].values:
                ("random frames already annotated. Skipping")
                return 10

        if self.back == True:
            try:
                a = self.annot_data.loc[self.annot_data['frame'] == start_frame]
                return list(self.BEHAVIOR_LABELS.keys())[
                    list(self.BEHAVIOR_LABELS.values()).index(a['pred'].to_string(index=False).strip())]
            except:
                return 9

        for pred in np.arange(start_frame, stop_frame + 1):
            if pred in self.annot_pickle['frame'].values or pred in self.annot_data['frame'].values:
                continue
            try:
                preds.append(self.pred_dict[pred])
            except:
                continue
        most_pred = stats.mode(preds, axis=None)
        if most_pred[0].size < 1:
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
            if self.back == False:
                self.annot_data = self.annot_data.append(
                    {'frame': frame, 'pred': self.BEHAVIOR_LABELS[int(self.det_pred)]}, ignore_index=True)
            else:
                self.annot_data.loc[self.annot_data['frame'] == frame, 'pred'] = self.BEHAVIOR_LABELS[int(self.det_pred)]


    def save_annotations_as_pickle(self):
        """
        Converts annotations to pickle file.
        :return:
        pickle file with columns [frame, pred]
        """
        self.annot_pickle_final = pd.concat([self.annot_pickle, self.annot_data])
        self.annot_pickle_final.sort_values(by='frame',inplace=True)
        self.annot_pickle_final.drop_duplicates(subset=['frame'],inplace=True,keep='last')
        if self.test_or_train == 'test':
            self.annot_pickle_final.to_pickle(self.pickle_rsync)
            subprocess.run("rsync --progress {} jbecke11@serrep6.clps.brown.edu:{}".format(
                self.pickle_rsync, self.pickle_path),
                           shell=True)
        else:
            self.annot_pickle_final.to_pickle(self.pickle_rsync)
            subprocess.run("rsync --progress {} jbecke11@serrep6.clps.brown.edu:{}".format(
                self.pickle_rsync, self.pickle_path ),
                           shell=True)

        a = pd.read_pickle(self.pickle_rsync)
        print(a)

        self.done_with_video()


    def done_with_video(self):
        """
        Asks if you are done with video and diplays percentage of video analyzed.
        Either moves video and csv file to done directory, or keeps in not_done directory.


        print("Updating Config File")

        final_test = {}
        test_vers_update = {}
        for file in glob.glob(self.pickle_path + "/test/*"):
            sample_total = {"drink": 0,
                          "groom": 0,
                          "eat": 0,
                          "hang": 0,
                          "sniff": 0,
                          "rear": 0,
                          "rest": 0,
                          "walk": 0,
                          "eathand": 0,
                          "none": 0}
            pred_sum = {}
            self.annot_pickle = pd.read_pickle(file)
            self.annot_pickle.sort_values(by='frame', inplace=True)
            self.annot_pickle = self.annot_pickle[self.annot_pickle.pred != 'none']
            test_beh = (self.annot_pickle)
            file = file[file.rfind('/') + 1:]
            test_experiment = (file[:file.find('_')])
            matches = re.finditer("_", file)
            matches_positions = [match.start() for match in matches]
            test_exp_version = (file[matches_positions[1] + 1:matches_positions[2]])

            frame_one = self.annot_pickle['frame'].iloc[0]
            frame_lst = self.annot_pickle['frame'].iloc[0]
            for index, row in self.annot_pickle.iterrows():
                if row['frame'] - frame_lst >1:
                    if frame_lst - frame_one >= 64:
                        pred_comp = self.annot_pickle.loc[self.annot_pickle['frame'] == frame_one]
                        pred_comp = pred_comp['pred'].to_string(index=False).strip()
                        for frame in np.arange(frame_one+1, frame_lst-63):
                            pred_lst = self.annot_pickle.loc[self.annot_pickle['frame'] == frame]
                            pred_lst = pred_lst['pred'].to_string(index=False).strip()
                            sample_total[pred_lst] += 1
                            if pred_comp != pred_lst:
                                if pred_comp in pred_sum:
                                    pred_sum[pred_comp] += 1
                                    pred_comp = self.annot_pickle.loc[self.annot_pickle['frame'] == frame]
                                    pred_comp = pred_comp['pred'].to_string(index=False).strip()
                                else:
                                    pred_sum[pred_comp] = 1
                                    pred_comp = self.annot_pickle.loc[self.annot_pickle['frame'] == frame]
                                    pred_comp = pred_comp['pred'].to_string(index=False).strip()

                        if pred_comp in pred_sum:
                            pred_sum[pred_comp] += 1

                        else:
                            pred_sum[pred_comp] = 1

                        frame_one = row['frame']
                        frame_lst = row['frame']
                    else:
                        frame_one = row['frame']
                        frame_lst = row['frame']
                else:
                    frame_lst= row['frame']

            if frame_lst - frame_one >= 64:
                pred_comp = self.annot_pickle.loc[self.annot_pickle['frame'] == frame_one]
                pred_comp = pred_comp['pred'].to_string(index=False).strip()
                for frame in np.arange(frame_one + 1, frame_lst - 63):
                    pred_lst = self.annot_pickle.loc[self.annot_pickle['frame'] == frame]
                    pred_lst = pred_lst['pred'].to_string(index=False).strip()
                    if pred_comp != pred_lst:
                        if pred_comp in pred_sum:
                            pred_sum[pred_comp] += 1
                            pred_comp = self.annot_pickle.loc[self.annot_pickle['frame'] == frame]
                            pred_comp = pred_comp['pred'].to_string(index=False).strip()
                        else:
                            pred_sum[pred_comp] = 1
                            pred_comp = self.annot_pickle.loc[self.annot_pickle['frame'] == frame]
                            pred_comp = pred_comp['pred'].to_string(index=False).strip()
                if pred_comp in pred_sum:
                    pred_sum[pred_comp] += 1

                else:
                    pred_sum[pred_comp] = 1

            test_update_params={}
            test_total = {"drink": [0,0],
                          "groom": [0,0],
                          "eat": [0,0],
                          "hang": [0,0],
                          "sniff": [0,0],
                          "rear": [0,0],
                          "rest": [0,0],
                          "walk": [0,0],
                          "eathand": [0,0],
                          "none": [0,0]}


            for key_test in pred_sum.keys():
                test_total[key_test][0] += pred_sum[key_test]

            for key_test in sample_total.keys():
                test_total[key_test][1] += sample_total[key_test]

            if test_exp_version in test_vers_update:
                if test_experiment in test_vers_update[test_exp_version]:
                    test_update_params[test_experiment] = test_total
                    for key_tests in test_update_params[test_experiment].keys():
                        test_vers_update[test_exp_version][test_experiment][key_tests][0]+=test_total[key_tests][0]
                        test_vers_update[test_exp_version][test_experiment][key_tests][1] += test_total[key_tests][1]

                else:
                    test_vers_update[test_exp_version][test_experiment]={0:0}
                    test_vers_update[test_exp_version][test_experiment] = test_total
            else:
                test_update_params[test_experiment] = test_total
                test_vers_update[test_exp_version] = test_update_params

        config_param = {"Boot Round": self.boot_round, "Main Path": self.main_path}
        final_test["Number of behaviors for Test Set"] = test_vers_update


        print("config file update done")
        with open(self.main_path+"/config.yaml", 'w') as file:
            documents = yaml.dump(config_param, file)
            documents = yaml.dump(final_test,file)

        file.close()
        """
        frame_total = self.video_length
        try:
            self.frames_analyzed.append(len(self.annot_data.index))
        except:
            pass
        analyzed_frames = len(self.annot_pickle_final)
        ndarray2 = np.full((640, 900, 3), 0, dtype=np.uint8)

        title_image3 = cv2.putText(ndarray2, "You have analyzed {} percent of video".format(
            (analyzed_frames / frame_total) * 100),
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
        while True:
            j = cv2.waitKey(0)
            if j == ord('s'):
                cv2.destroyAllWindows()
                sys.exit()
            elif j == ord('m'):
                cv2.destroyAllWindows()
                os.rename(self.video_file,
                          self.main_path + '/videos_done' + self.video_file[self.video_file.rfind('/'):])
                sys.exit()

def main():

    parser = argparse.ArgumentParser(description="Add main path and frame length for video loop")
    parser.add_argument("-mp", "-main_path", help="Directory where you want all files associated with artemis annotations saved. This will create a folder called Annot whcih will hold all files,"
                                                  "Different experiments can be housed in separate folders under different Annot folder")
    parser.add_argument("-rp", "-rsync_path",
                        help="Directory where you want all files associated with artemis annotations saved. This will create a folder called Annot whcih will hold all files,"
                             "Different experiments can be housed in separate folders under different Annot folder")
    parser.add_argument("-f", "-frame_length", const=100, type=int, nargs="?", default=100, help="number of frames to analyze in each loop: default is 100 frames")
    parser.add_argument("-ps", "-playback_speed", const=1, type=float, nargs="?", default=1,
                        help="playback speed of interval. Higher number speeds up playback. Lower number slows playback ")
    args = parser.parse_args()
    return args.mp, args.rp, args.f, args.ps


if __name__ == "__main__":

    mp, rp, f, ps = main()
    rp = rp + "Annot"
    mp = mp + "Annot"

    artemis = show_prediction(mp, rp)

    artemis.show_intro()
    frame = int(artemis.load_video_organize_dir())
    artemis.determine_last_frame()
    artemis.loop_video(frame, f, ps)

#1. DOUBLE CHECK FILES ON ARTEMIS SIDE IN CCV (LAST 3 SYNCED) AND SYNC 2 THAT WERE JUST ANALYZED (inference test and results)
#Work on appending to config.yaml from bootstrap code side!
#Figure out how to do playback when csv file is present and dsplay pred!
#add messages for esceptions
#cut down onf code

#arguents to add for commercialization
#2. ability to start back up with same configurations as last analysis time (cvs/nocsv, and video)



