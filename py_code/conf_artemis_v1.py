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
import yaml
from sklearn.metrics import confusion_matrix

class conf_matrix_artemis():

    def __init__(self, path_to_video, path_to_csv):
        self.path_to_video = path_to_video
        self.path_to_csv = path_to_csv
        with open('../Annot/config.yaml') as config:
            data = yaml.load(config, Loader=yaml.FullLoader)
            self.boot_round = data['Boot Round']
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
            9: "no pred",
            10: "No annotation"
        }
        self.BEHAVIOR_NAMES = {
            "drink":0,
            "groom":2,
            "eat":1,
            "hang":3,
            "sniff":4,
            "rear":5,
            "rest":6,
            "walk":7,
            "eathand":8,
            "none":9,
        }

        self.csv_results = {}
        self.num_right = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        self.total_analyzed = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        self.accuracy_annotations = np.zeros((9, 9))
        self.confusion_matrix = np.zeros((9, 9))
        self.nooo = np.empty((1, 9))
        self.nooo[:] = np.NaN
        self.accuracy_annotations = np.zeros(shape=(9, 9))
        self.confusion_matrix = np.zeros(shape=(9, 9))
        self.analyze_csv = []
        self.analyze_pickle = []
        self.y_true = np.zeros(9)
        self.y_pred = np.zeros(9)
        print(self.y_pred)
        print(self.y_true)

    def check_load_csv(self):
        root = tk.Tk()
        root.withdraw()
        self.csv_file = askdirectory(initialdir="Annot/csv_not_done",
                                     title="Select CSV folder")  # show an "Open" dialog box and return the path to
        # the selected file
        self.pickle_path = askdirectory(initialdir="Annot/pickle_files", title="Select pickle folder")

        # TODO: First iterate through pkl test files. If there is a matching csv file, append both to respectile
        #  list. This will guarantee that element at each index in list corresponds to respective element at index in
        #  other list.

        set_of_csv = set()
        set_of_pkl = set()

        try:
            for csv in glob.glob(self.csv_file + '/*.csv'):
                # We clean the csv name to find just the file name:
                #   csv_not_done/abc.csv --> abc
                file_name = csv.replace(self.csv_file, '').replace('.csv', '')
                set_of_csv.add(file_name)
        except:
            print('No CSV file in directory. Transfer some and run again')

        # Suffix for rebuilding pickle name.
        pickle_suffix = ''
        try:
            for picklefile in glob.glob(self.pickle_path + '/*.p'):
                file_name = picklefile.replace(self.pickle_path, '')
                # Saves the suffix past the last '_' to rebuild pickle name later.
                pickle_suffix = file_name[file_name.rfind('_'):]
                # Finds the last occurence of '_', and takes the string up to that.
                file_name = file_name[:file_name.rfind('_')]
                # test/abc_test.p --> abc
                set_of_pkl.add(file_name)
        except:
            print('No Pickle file in directory. Transfer some and run again')

        common_files = list(set_of_pkl.intersection(set_of_csv))
        # We rebuild list of csv and pickles from this intersection.
        for file in common_files:
            csv_name_rebuilt = self.csv_file + file + '.csv'
            self.analyze_csv.append(csv_name_rebuilt)
            pickle_name_rebuilt = self.pickle_path + file + pickle_suffix
            self.analyze_pickle.append(pickle_name_rebuilt)



    def analyze_csv_pickle(self):

        for i in np.arange(0,len(self.analyze_pickle)):

            csv_file = self.analyze_csv[i]
            pickle_file = self.analyze_pickle[i]
            csv_data = pd.read_csv(csv_file, names=['frame','pred'])
            csv_data.sort_values(by='frame',inplace=True)
            csv_data.drop_duplicates(subset=['frame'],inplace=True,keep='last')
            annotations = pd.read_pickle(pickle_file)
            annotations = annotations[annotations.pred != "none"]
            annotations.sort_values(by='frame', inplace=True)
            annotations.drop_duplicates(subset=['frame'], inplace=True, keep='last')
            start_frame = annotations['frame'].iloc[0]

            for index,row in annotations.iterrows():
                end_frame = (row['frame'])
                if end_frame - start_frame == 10:
                    ten_frames_annotations = []  # make list to store last 10 annotations and predictions
                    ten_frames_predictions = []
                    for frame in np.arange(start_frame, end_frame):
                        a = annotations[annotations['frame'] == frame]
                        b = csv_data[csv_data['frame']==frame]
                        ten_frames_annotations.append(self.BEHAVIOR_NAMES[(a['pred'].to_string(index=False).strip())])
                        ten_frames_predictions.append(b['pred'].to_string(index=False).strip())

                    most_annotation = stats.mode(ten_frames_annotations, axis=None)
                    most_prediction = stats.mode(ten_frames_predictions, axis=None)
                    self.total_analyzed[int(most_annotation[0])] +=1
                    self.accuracy_annotations[int(most_annotation[0]), int(most_prediction[0])] += 1
                    start_frame = end_frame

    def analyze_csv_pickle_sk(self):
        for i in np.arange(0, len(self.analyze_pickle)):

            csv_file = self.analyze_csv[i]
            
            pickle_file = self.analyze_pickle[i]
            csv_data = pd.read_csv(csv_file, names=['frame', 'pred'])
            csv_data.sort_values(by='frame', inplace=True)
            csv_data.drop_duplicates(subset=['frame'], inplace=True, keep='last')
            annotations = pd.read_pickle(pickle_file)
            annotations = annotations[annotations.pred != "none"]
            annotations.sort_values(by='frame', inplace=True)
            annotations.drop_duplicates(subset=['frame'], inplace=True, keep='last')
            start_frame = annotations['frame'].iloc[0]
            ten_frames_annotations = []  # make list to store last 10 annotations and predictions
            ten_frames_predictions = []
            for index, row in annotations.iterrows():
                frame = int(row['frame'])
                a = annotations[annotations['frame'] == frame]
                b = csv_data[csv_data['frame'] == frame]
                a = self.BEHAVIOR_NAMES[(a['pred'].to_string(index=False).strip())]
                b = b['pred'].to_string(index=False).strip()

                self.y_pred[int(b)] += 1
                self.y_true[int(a)] +=1


        conf_matrix = confusion_matrix(self.y_true,self.y_pred)
        print(self.y_true)
        print(self.y_pred)
        print(conf_matrix)

        for i in np.arange(len(self.accuracy_annotations)):
            norm_sum = np.sum(self.accuracy_annotations[i][0:])
            if norm_sum != 0:
                array_norm = self.accuracy_annotations[i][0:] / norm_sum
                self.confusion_matrix[i][0:] = (array_norm)


    def compute_confusion_matrix(self):

        csv_data_df = []
        for csv in self.analyze_csv:
            data = pd.read_csv(csv, names=['frame', 'pred']).drop_duplicates(subset='frame')
            csv_data_df.append(data)
        # Dataframe of  all csv data.

        pkl_data_df = []
        for pickle in self.analyze_pickle:
            data = pd.read_pickle(pickle)
            data['pred'] = data['pred'].apply(lambda x: self.BEHAVIOR_NAMES.get(x))
            pkl_data_df.append(data)

        y_pred = []
        y_true = []

        for csv_, pkl in zip(csv_data_df, pkl_data_df):
            csv_data_for_pkl = csv_.loc[csv_['frame'].isin(pkl['frame'])]
            y_pred.append(csv_data_for_pkl['pred'])
            y_true.append(pkl['pred'])

        # Labels array of dimensions (n_classes)
        labels = [mapping[0] for mapping in list(self.BEHAVIOR_LABELS.items())]
        y_pred = pd.concat(y_pred)
        y_true = pd.concat(y_true)
        self.confusion_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels,
                                                 normalize='true')

    def build_heatmap(self):
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
    a = conf_matrix_artemis('../Annot', '../Annot')
    a.check_load_csv()
    a.compute_confusion_matrix()
    a.analyze_csv_pickle()
    a.build_heatmap()
    a.analyze_csv_pickle_sk()
