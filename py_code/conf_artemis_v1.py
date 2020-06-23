import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sn
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import simpledialog
import pandas as pd
import re

class conf_matrix_artemis():

    def __init__(self, path_to_video, path_to_csv):
        self.path_to_video = path_to_video
        self.path_to_csv = path_to_csv
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

        self.csv_results = {}
        self.num_right = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        self.total_analyzed = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        self.accuracy_annotations = np.zeros((9,9))
        self.confusion_matrix = np.zeros((9,9))
        self.nooo = np.empty((1,9))
        self.nooo[:]= np.NaN
        self.csv_name = []
        self.pickle_name = []
        self.accuracy_annotations = np.zeros(shape=(9,9))
        self.confusion_matrix = np.zeros(shape=(9,9))

    def check_load_csv(self):
        root = tk.Tk()
        root.withdraw()
        self.analyze_csv= []
        self.analyze_pickle = []
        i = 0
        self.boot_round = simpledialog.askstring(title="Test",
                                                 prompt="What bootstrap round is this? (0,2, etc; or type all")
        self.csv_file = askdirectory(initialdir="/home/jordan/Desktop/nihgpppipe/Annot/csv_not_done",
                                        title="Select CSV folder")  # show an "Open" dialog box and return the path to the selected file
        self.pickle_path = askdirectory(initialdir="/home/jordan/Desktop/nihgpppipe/Annot/pickle_files",title="Select pickle folder")
        try:
            for csvfile in glob.glob(self.csv_file + '/*.csv'):
                self.csv_name.append(csvfile[csvfile.rfind('/')+1:])
        except:
            print('No CSV file in directory. Transfer some and run again')

        try:
            for picklefile in glob.glob(self.pickle_path + '/*.p'):
                if self.boot_round == picklefile[-3]:
                    self.pickle_name.append(picklefile[picklefile.rfind('/')+1:])
                elif self.boot_round.lower() == 'all':
                    self.pickle_name.append(picklefile[picklefile.rfind('/')+1:])

        except:
            print('No Pickle file in directory. Transfer some and run again')
        for file in self.pickle_name:
            for csv in self.csv_name:
                if re.match(file[:file.rfind('_')]+".*",csv):
                    self.analyze_pickle.append(file)
        print(self.analyze_pickle)

    def analyze_csv_pickle(self):
        for file in self.analyze_pickle:
            csv_data = pd.read_csv(self.csv_file+'/{}'.format(file[:file.rfind('_')])+'.csv', names=[0,1])
            annotations = pd.read_pickle(self.pickle_path+"/"+file)
            start_frame = annotations['frame'][1]
            for index,row in annotations.iterrows():
                end_frame = (row['frame'])
                if end_frame - start_frame == 10:
                    ten_frames_annotations = []  # make list to store last 10 annotations and predictions
                    ten_frames_predictions = []
                    for frame in np.arange(start_frame, end_frame):
                        try:
                            ten_frames_annotations.append(annotations['pred'][frame])
                            ten_frames_predictions.append(csv_data[1][frame])
                        except:
                            continue

                    most_annotation = stats.mode(ten_frames_annotations, axis=None)
                    most_prediction = stats.mode(ten_frames_predictions, axis=None)

                    # total_analyzed[int(most_annotation[0])] +=1
                    try:
                        self.accuracy_annotations[most_annotation[0], most_prediction[0]] += 1
                    except:
                        continue

                    start_frame = end_frame
        for i in np.arange(len(self.accuracy_annotations)):
            norm_sum = np.sum(self.accuracy_annotations[i][0:])
            if norm_sum != 0:
                array_norm = self.accuracy_annotations[i][0:] / norm_sum
                self.confusion_matrix[i][0:] = (array_norm)
        plt.figure(figsize=(9, 9))
        beh = ['drink', 'eat', 'groom',
               'hang', 'sniff', 'rear', 'rest',
               'walk', 'eathand']
        graph = sn.heatmap(self.confusion_matrix, xticklabels=beh, yticklabels=beh, annot=True, cmap="YlGnBu")
        plt.yticks(rotation=0)
        plt.ylabel('Ground Truth')
        plt.xlabel('Predictions')
        plt.show()

if __name__ == "__main__":
    a = conf_matrix_artemis('/home/jordan/Desktop/nihgpppipe/Annot','/home/jordan/Desktop/nihgpppipe/Annot')
    a.check_load_csv()
    a.analyze_csv_pickle()