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
import argparse
from operator import add


class conf_matrix_artemis():

    def __init__(self, config_path, annotation_path, prediction_path):
        self.config_path = config_path + "config.yaml"
        self.annotation_path = annotation_path
        self.prediction_path = prediction_path
        with open(self.config_path) as config:
            data = yaml.load(config, Loader=yaml.FullLoader)
            self.boot_round = data['Boot Round']
            self.new_test_set_beh = data["Number of behaviors for Test Set"]["new"]
            self.old_test_set_beh = data["Number of behaviors for Test Set"]["old"]
            print(self.new_test_set_beh)
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
        # TODO: First iterate through pkl test files. If there is a matching csv file, append both to respectile
        #  list. This will guarantee that element at each index in list corresponds to respective element at index in
        #  other list.
        set_of_csv = set()
        set_of_pkl = set()
        try:
            for csv in glob.glob(self.prediction_path + '*.csv'):
                # We clean the csv name to find just the file name:
                #   csv_not_done/abc.csv --> abc
                file_name = csv.replace(self.prediction_path, '').replace('.csv', '')
                set_of_csv.add(file_name)
        except:
            print('No CSV file in directory. Transfer some and run again')
        # Suffix for rebuilding pickle name.
        pickle_suffix = ''
        try:
            for picklefile in glob.glob(self.annotation_path + '*.p'):
                file_name = picklefile.replace(self.annotation_path, '')
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
            csv_name_rebuilt = self.prediction_path + file + '.csv'
            self.analyze_csv.append(csv_name_rebuilt)
            pickle_name_rebuilt = self.annotation_path + file + pickle_suffix
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

    def unpack_metrics(self, drink=[0, 0], eat=[0, 0], groom=[0, 0],
                       hang=[0, 0], sniff=[0, 0], rear=[0, 0], rest=[0, 0], walk=[0, 0], eathand=[0, 0], none=[0, 0]):
        return {"drink": drink, "eat": eat, "groom": groom, "hang": hang,
                "sniff": sniff, "rear": rear, "rest": rest, "walk": walk, "eathand": eathand}

    def calculate_metrics(self):
        # create empyty dictionaries to append bouts and samples to
        self.new_sum = {"drink": [0, 0],
                        "eat": [0, 0],
                        "groom": [0, 0],
                        "hang": [0, 0],
                        "sniff": [0, 0],
                        "rear": [0, 0],
                        "rest": [0, 0],
                        "walk": [0, 0],
                        "eathand": [0, 0]}
        self.old_sum = {"drink": [0, 0],
                        "eat": [0, 0],
                        "groom": [0, 0],
                        "hang": [0, 0],
                        "sniff": [0, 0],
                        "rear": [0, 0],
                        "rest": [0, 0],
                        "walk": [0, 0],
                        "eathand": [0, 0]}

        self.bout = pd.DataFrame()
        self.sample = pd.DataFrame()

        for key in self.new_test_set_beh.keys():
            a = self.unpack_metrics(**self.new_test_set_beh[key])
            for beh in a.keys():
                self.new_sum[beh] = list(map(add, self.new_sum[beh],a[beh]))

        for key in self.old_test_set_beh.keys():
            a = self.unpack_metrics(**self.old_test_set_beh[key])
            for beh in a.keys():
                self.old_sum[beh] = list(map(add, self.old_sum[beh],a[beh]))

        self.new_sum = pd.DataFrame.from_dict(self.new_sum,orient="index")
        #self.new_sum["x"] = ["bouts","samples"]
        self.old_sum = pd.DataFrame.from_dict(self.old_sum,orient="index")
        self.bout["New Video"] = self.new_sum[0]
        self.bout["Old Video"] = self.old_sum[0]
        self.sample["New Video"] = self.new_sum[1]
        self.sample["Old Video"] = self.old_sum[1]



    def build_metrics(self):
        fig, axes = plt.subplots(1,2,figsize=(10,7))

        a = self.bout.plot.bar(ax=axes[0],title="Bouts for each behavior", legend=False)
        b = self.sample.plot.bar(ax=axes[1],title="Samples(frames) for each behavior",
                                 yticks=np.linspace(0,20000,2))

        plt.show()




def main():
    parser = argparse.ArgumentParser(
        description="Add path to config file, path to annotations and path to predictions.")
    parser.add_argument("-mp", "-main_path",
                        help="path to Annot folder in experiment you want to analyze")
    args = parser.parse_args()
    return args.mp


if __name__ == "__main__":
    # main_path = main()
    config_path = "/home/jordan/Desktop/andrew_nih/Annot/"
    annotation_path = "/home/jordan/Desktop/andrew_nih/Annot/pickle_files/test/"
    prediction_path = "/home/jordan/Desktop/andrew_nih/Annot/csv_not_done/"
    a = conf_matrix_artemis(config_path, annotation_path, prediction_path)
    # a.check_load_csv()
    # a.compute_confusion_matrix()
    # a.build_heatmap()
    d = {"eat": [1, 2], "drink": [5, 3]}
    a.calculate_metrics()
    a.build_metrics()

"""
NEXT STEPS:
2. Make 2 more confusion matrices: 1) confusion matrix on old videos, 2) confusion matrix on new videos
3. Output nice graphic that details how many examples you have of each behavior (can use cnfig file for this)
"""
