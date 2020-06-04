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
from collections import defaultdict


class show_prediction():

    def __init__(self, main_path, boot_round, new_annot):
        """
        Initialize variables for script and dictionary with
        keys to behavior label. You can set up this
        dictionary however you like - pickle file with save
        keys to frame, while values will be displayed on video.
        :param main_path:
       :
        """
        self.back = False
        self.main_path = main_path
        self.boot_round = boot_round
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.new_annot = new_annot
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
        try:
            os.makedirs(self.main_path + "/videos_to_analyze")
        except:
            pass
        try:
            os.makedirs(self.main_path + '/videos_not_done')

        except:
            if len(os.listdir(self.main_path + '/videos_not_done')) == 0:
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
            os.makedirs(self.main_path + '/csv_not_done')
        except:
            if len(os.listdir(self.main_path + '/csv_not_done')) == 0:
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
            os.makedirs(self.main_path + '/pickle_files/train')
            os.makedirs(self.main_path + '/pickle_files/test')
        except:
            pass
        root = tk.Tk()
        root.withdraw()
        self.test_or_train = simpledialog.askstring(title="Test",
                                                 prompt="Add annotations to test or train dataset:")

        self.pickle_path = self.main_path + "/pickle_files"

        self.csv_file = askopenfilename(initialdir= self.main_path + "/csv_not_done",
                                        title="Select CSV file")  # show an "Open" dialog box and return the path to the selected file
        self.video_file = askopenfilename(initialdir=self.main_path + "/videos_not_done",
                                          title="Select VIDEO file")

        if self.csv_file:
            self.predictions = pd.read_csv(self.csv_file, names=['frame', 'pred'])
        self.cap = cv2.VideoCapture(self.video_file)

        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_arr = np.arange(80,(self.video_length+1))
        self.total_frames = pd.DataFrame(data=frame_arr, columns=['frame'])

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
            self.non_analyzed_frames = pd.concat([self.total_frames,self.exp_frames_analyzed,self.exp_frames_analyzed],sort=True).drop_duplicates(subset=['frame'],keep=False)

        except:
            self.non_analyzed_frames = self.total_frames
            pass
        try:
            self.pred_dict = pd.Series(self.predictions.pred.values,index=self.predictions.frame).to_dict()
            self.prediction_state = True
        except:
            pass

        if self.test_or_train == 'test':
            try:
                self.annot_pickle = pd.read_pickle(
                    self.pickle_path + "/test" + self.video_file[self.video_file.rfind('/'):-4] + '_test{}.p'.format(self.boot_round))
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
                    self.pickle_path + "/train" + self.video_file[self.video_file.rfind('/'):-4] + '_boot{}.p'.format(self.boot_round))
                self.annot_pickle.sort_values(by='frame', inplace=True)
                self.annot_pickle.drop_duplicates(subset=['frame'])
                pickl_pres = True
            except:
                self.annot_pickle = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
                pickl_pres = False
                pass
        self_start = (self.non_analyzed_frames['frame'].iloc[0])
        print(f"Your current pickle file has {len(self.annot_pickle)} frames annotated")

        return  self_start

    def loop_video(self, start_frame=80, interval=100, playback_speed = 1):
        """
        Loops over video with gui. This is where you update or confirm annotations.
        :param start_frame:
        :param interval:
        :return:
        Appends annotation to pandas dataframe
        """
        print(f"You have analyzed {len(self.annot_data)} frames in this session")
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
                    print("Frames already annotated in last save. Continuing to next interval ")
                    self.loop_video((self.start_frame + interval), interval)
                frame_pred = cv2.putText(frame, "Current: Frame " + str(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) +  "   Pred: " +
                                         self.BEHAVIOR_LABELS[int(self.det_pred)], (5, 25),
                                         cv2.FONT_HERSHEY_DUPLEX, 0.75,
                                         (60, 76, 231), 1, cv2.LINE_AA)
                cv2.namedWindow('image')
                cv2.moveWindow('image', 0, 0)
                cv2.imshow('image', frame)
                cv2.waitKey(int((1 / (interval * playback_speed))*1000))
            else:
                if self.annotating == True:
                    frame_pred = cv2.putText(frame, "Loop Done.", (5, 50),
                                             cv2.FONT_HERSHEY_DUPLEX, 0.75,
                                             (60, 76, 231), 1, cv2.LINE_AA)
                    cv2.imshow('image', frame)
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
                        self.forward = False
                        self.back = False
                        self.loop_video((self.start_frame + interval), interval, playback_speed)
                    elif k & 0xFF == ord('\x1b'):
                        self.annotating = False
                        cv2.destroyAllWindows()
                        self.det_pred = None
                        self.save_annotations_as_pickle()
                    elif k & 0xFF == ord(' '):
                        self.loop_video(self.start_frame, interval, playback_speed)
                    elif k & 0xFF == ord('Q'):
                        self.back = True
                        self.loop_video(self.start_frame - interval, interval, playback_speed)
                    elif k & 0xFF == ord('S'):
                        self.forward = True
                        self.loop_video((self.start_frame + interval), interval, playback_speed)
                    elif k & 0xFF == ord('1') or k & 0xFF == ord('2') or k & 0xFF == ord('3') or k & 0xFF == ord(
                            '4') or k & 0xFF == ord('5') or k & 0xFF == ord('6') or k & 0xFF == ord(
                            '7') or k & 0xFF == ord('8') or k & 0xFF == ord('9'):
                        self.det_pred = int(chr(k))
                        self.update_annotations()
                        self.forward = False
                        self.back = False
                        self.loop_video((self.start_frame + interval), interval, playback_speed)
                    elif k & 0xFF == ord('y'):
                        self.update_annotations()
                        self.forward = False
                        self.back = False
                        self.loop_video((self.start_frame + interval), interval, playback_speed)

    def     determine_prediction(self, start_frame, stop_frame):
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
            if start_frame not in self.annot_pickle['frame'].values and start_frame not in self.annot_data['frame'].values:
                return 9
            elif self.back == True:
                a = self.annot_data.loc[self.annot_data['frame'] == start_frame]
                return list(self.BEHAVIOR_LABELS.keys())[list(self.BEHAVIOR_LABELS.values()).index(a['pred'].to_string(index=False).strip())]
            elif self.forward == True:
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
            self.annot_pickle_final.to_pickle(self.pickle_path + "/test"+self.video_file[self.video_file.rfind('/'):-4] + '_test{}.p'.format(self.boot_round))
        else:
            self.annot_pickle_final.to_pickle(self.pickle_path +"/train"+ self.video_file[self.video_file.rfind('/'):-4] + '_boot{}.p'.format(self.boot_round))
        self.done_with_video()

    def done_with_video(self):
        """
        Asks if you are done with video and diplays percentage of video analyzed.
        Either moves video and csv file to done directory, or keeps in not_done directory.
        """
        update_beh = []
        test_update_beh = []
        update_params = {}
        test_update_params = {}
        boot_update_params = {}
        final_test = {}
        experiments = []
        test_experiments = []
        for file in glob.glob(self.pickle_path + "/train/*"):
            self.annot_pickle = pd.read_pickle(file)
            self.annot_pickle.sort_values(by='frame',inplace=True)
            update_beh.append(self.annot_pickle)
            file = file[file.rfind('/')+1:]
            experiments.append(file[:file.find('_')])
        for file in glob.glob(self.pickle_path + "/test/*"):
            self.annot_pickle = pd.read_pickle(file)
            self.annot_pickle.sort_values(by='frame',inplace=True)
            test_update_beh.append(self.annot_pickle)
            file = file[file.rfind('/')+1:]
            test_experiments.append(file[:file.find('_')])

        for i in np.arange(0,len(update_beh)):
            beh_total = {"drink": 0,
                         "groom": 0,
                         "eat": 0,
                         "hang": 0,
                         "sniff": 0,
                         "rear": 0,
                         "rest": 0,
                         "walk": 0,
                         "eathand": 0,
                         "none": 0}
            behavior_count = update_beh[i]["pred"].value_counts().to_dict()

            for key in behavior_count.keys():
                beh_total[key] += behavior_count[key]

            if experiments[i] in update_params:
                for key in beh_total.keys():
                    update_params[experiments[i]][key] += beh_total[key]
            else:
                update_params[experiments[i]] = beh_total

        for i in np.arange(0,len(test_update_beh)):
            test_total = {"drink": 0,
                          "groom": 0,
                          "eat": 0,
                          "hang": 0,
                          "sniff": 0,
                          "rear": 0,
                          "rest": 0,
                          "walk": 0,
                          "eathand": 0,
                          "none": 0}

            test_count = test_update_beh[i]["pred"].value_counts().to_dict()

            for key_test in test_count.keys():
                test_total[key_test] += test_count[key_test]

            if test_experiments[i] in test_update_params:
                for key in beh_total.keys():
                    test_update_params[test_experiments[i]][key_test] += test_total[key_test]
            else:
                test_update_params[test_experiments[i]] = test_total

        config_param = {"Boot Round": self.boot_round, "Main Path": self.main_path}
        boot_update_params["Number of behaviors for Boot {}".format(self.boot_round)] = update_params
        final_test["Number of behaviors for Test Set"] = test_update_params


        with open(self.main_path+"/config.yaml", 'w') as file:
            documents = yaml.dump(config_param, file)
            documents = yaml.dump(final_test,file)
            documents = yaml.dump(boot_update_params,file)

        file.close()

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
        title_image3 = cv2.putText(ndarray2, "Press s to keep video for more analysis",
                                   (20, 250), self.font,
                                   0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.namedWindow("Ins3")
        cv2.imshow("Ins3", ndarray2)
        while True:
            j = cv2.waitKey(0)
            if j == ord('s'):
                sys.exit()

def main():

    parser = argparse.ArgumentParser(description="Add main path and frame length for video loop")
    parser.add_argument("-mp", "-main_path", help="Directory where you want all files associated with artemis annotations saved. This will create a folder called Annot whcih will hold all files,"
                                                  "Different experiments can be housed in separate folders under different Annot folder")
    parser.add_argument("-f", "-frame_length", const=100, type=int, nargs="?", default=100, help="number of frames to analyze in each loop: default is 100 frames")
    parser.add_argument("-ps", "-playback_speed", const=1, type=float, nargs="?", default=1,
                        help="playback speed of interval. Higher number speeds up playback. Lower number slows playback ")
    args = parser.parse_args()
    return args.mp, args.f, args.ps

if __name__ == "__main__":

    mp, f, ps = main()
    mp = mp + "Annot"
    if os.path.exists(mp + "/config.yaml"):
        with open(mp + "/config.yaml") as file:
            config_param = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
            try:
                artemis = show_prediction(mp,config_param["Boot Round"],False)
            except:
                print("Need to add main path where CSV,video and pickle files will be held.\n"
                      "A folder called Annot will be save here and subsequent folders holding CSV, Video, and Pickle files will also be created or used.\n"
                      "Use -mp followed by path when calling script in terminal.")
                sys.exit(1)
    else:
        artemis = show_prediction(mp, 1,True)

    artemis.show_intro()
    artemis.load_video_organize_dir()
    artemis.loop_video(artemis.determine_last_frame(), f, ps)

#Work on appending to config.yaml from bootstrap code side!
#add messages for esceptions
#cut down onf code

#arguents to add for commercialization
#2. ability to start back up with same configurations as last analysis time (cvs/nocsv, and video)



