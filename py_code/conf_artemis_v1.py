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
            1: "eat",
            2: "groom",
            3: "hang",
            4: "sniff",
            5: "rear",
            6: "rest",
            7: "walk",
            8: "eathand",
        }
        self.BEHAVIOR_NAMES = {
            "drink": 0,
            "eat": 1,
            "groom": 2,
            "hang": 3,
            "sniff": 4,
            "rear": 5,
            "rest": 6,
            "walk": 7,
            "eathand": 8,
            "none": 9,
        }

        self.analyze_csv = []
        self.analyze_pickle = []


    def check_load_csv(self):
        root = tk.Tk()
        root.withdraw()
        self.csv_file = askdirectory(initialdir="/home/jordan/Desktop/andrew_nih/Annot/csv_not_done",
                                     title="Select CSV folder")  # show an "Open" dialog box and return the path to
        # the selected file
        self.pickle_path = askdirectory(initialdir="/home/jordan/Desktop/andrew_nih/Annot/pickle_files/test", title="Select pickle folder")

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

    def compute_confusion_matrix(self):

        csv_data_df = []
        for csv in self.analyze_csv:
            data = pd.read_csv(csv, names=['frame', 'pred']).drop_duplicates(subset='frame')
            data = data[data.pred != 9]
            csv_data_df.append(data)
        # Dataframe of  all csv data.

        pkl_data_df = []
        for pickle in self.analyze_pickle:
            data = pd.read_pickle(pickle)
            data['pred'] = data['pred'].apply(lambda x: self.BEHAVIOR_NAMES.get(x))
            data = data[data['pred'] != 9]
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
        print(y_pred)
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
    a.build_heatmap()
    a.analyze_csv_pickle_sk()
