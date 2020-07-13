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
import h5py
import hdfdict
from collections import defaultdict
import csv
from sklearn.metrics import balanced_accuracy_score

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
            9: "none"
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
            "eathand":8,
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
            for picklefile in glob.glob(self.annotation_path + '*.h5'):
                file_name = picklefile.replace(self.annotation_path, '').replace('.mp4_annoated.h5','')

                # Saves the suffix past the last '_' to rebuild pickle name later.
                pickle_suffix = file_name[file_name.rfind('_'):]
                # Finds the last occurence of '_', and takes the string up to that.
                #file_name = file_name[:file_name.rfind('_')]
                # test/abc_test.p --> abc
                set_of_pkl.add(file_name)
        except:
            print('No Pickle file in directory. Transfer some and run again')
        common_files = list(set_of_pkl.intersection(set_of_csv))
        # We rebuild list of csv and pickles from this intersection.
        for file in common_files:
            csv_name_rebuilt = self.prediction_path + file
            self.analyze_csv.append(csv_name_rebuilt)
            pickle_name_rebuilt = self.annotation_path + file + '.mp4_annoated.h5'
            print(pickle_name_rebuilt)
            self.analyze_pickle.append(pickle_name_rebuilt)


    def get_predicted_true_labels(self, csv_data, pickle_data):
        csv_data_df = []
        for csv in csv_data:
            csv = csv + '.csv'
            with open(csv, 'rb') as f:
                preds = f.readlines()
                preds = [str(x).split(',')[-1].strip('\n\r') for x in preds]
                preds = [x[0] for x in preds]
                preds = [int(x[0]) for x in preds]

            csv_data_df.append(preds)

        # Dataframe of  all csv data.
        pkl_data_df = []
        for pickle in csv_data:
            pickle = pickle.replace(self.prediction_path, self.annotation_path)
            pickle = pickle + '.mp4_annoated.h5'
            data = h5py.File(pickle, 'r')
            gt = data.get('labels')
            gts = ['none'.encode() if x.decode('utf-8') == 'dig' or x.decode('utf-8') == 'body-turn' or
                   x.decode('utf-8') == 'jump' else x for x in gt]
            gts = [self.BEHAVIOR_NAMES[x.decode('utf-8')] for x in gts]
            print(len(gts))
            new_gts = gts[80:-80]
            pkl_data_df.append(new_gts)

        y_pred = [items for sublist in csv_data_df for items in sublist]
        y_true = [items for sublist in pkl_data_df for items in sublist]

        y_pred = self.slackify(y_true,y_pred)
        print(len(y_pred))
        print(len(y_true))
        return y_pred, y_true, self.balanced_accuracy(y_true,y_pred)

    def balanced_accuracy(self,y_true,y_pred):
        return balanced_accuracy_score(y_true,y_pred)

    def slackify(self,y_true, y_pred, slack=20):
        y_pred_slack = []
        for i, yp in enumerate(y_pred):
            # print(int(max(0,i-slack/2)),int(min(i+1+slack/2, len(y_true))))
            if yp in y_true[int(max(0, i - slack / 2)): int(min(i + 1 + slack / 2, len(y_true)))]:
                y_pred_slack.append(y_true[i])
            else:
                y_pred_slack.append(yp)
        return y_pred_slack

    def compute_confusion_matrix(self, csv=None, pkl=None):
        """
        :param csv: optional argument of list of csvs.
        :param pkl:
        :return:
        """
        csv_data = csv
        pkl_data = pkl
        if csv is None or pkl is None:
            csv_data = self.analyze_csv
            pkl_data = self.analyze_pickle

        # Labels array of dimensions (n_classes)
        labels = [mapping[0] for mapping in list(self.BEHAVIOR_LABELS.items()) if mapping[1] != 'none']

        y_pred, y_true, bal_acc = self.get_predicted_true_labels(csv_data, pkl_data)

        conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels,
                                       normalize='true')

        conf_matrix = np.round(conf_matrix,decimals=2)

        if csv is None or pkl is None:
            self.confusion_matrix = conf_matrix

        return conf_matrix, bal_acc

    def return_old_new(self, csv=None, pkl=None):
        """
        :param version: string of 'old' or 'new'. Raises exception if not either 'old' or 'new'.
        If string is 'old', returns filenames which contain 'old' as 3rd slot delineated by underscores, after
        file path has been cleaned from file path.
        :param csv: List of strings representing csv files to filter by version.
        :param pkl: List of strings representing pkl files to filter by version.
        :return: Tuple of (csv, pkl), each being list of strings, each element corresponding to old/new file.
                Includes file path in each element.
        """
        csv_files = csv
        pkl_files = pkl

        if csv is None or pkl is None:
            csv_files = self.analyze_csv
            pkl_files = self.analyze_pickle

        # TODO: Currently, each csv/pkl file in function arguments must contain self.csv_path or self.pickle_path
        #  respectively for this function to work. This is because we first clean up the strings by removing the
        #  file path prefix. This needs to be changed so that we split on some character that delineates the end
        #  of the path prefix and the start of the actual file name.

        csv_data = [csv.replace(self.prediction_path, '') for csv in csv_files]
        pkl_data = [pkl.replace(self.annotation_path, '') for pkl in pkl_files]


        old_csv = [self.prediction_path + csv for csv in csv_data]
        old_pkl = [self.annotation_path + pkl for pkl in pkl_data]
        return old_csv, old_pkl


        raise Exception('Error: version must be \'old\' or \'new\'.')

    def build_old_new_both_heatmap(self):
        old_csv, old_pkl = self.return_old_new()


        old_confusion, bal_acc = self.compute_confusion_matrix(csv=old_csv, pkl=old_pkl)




        beh = ['drink', 'eat', 'groom',
               'hang', 'sniff', 'rear', 'rest',
               'walk', 'eathand']

        fig = plt.figure(figsize=(12,12))


        ax0 = fig.add_subplot(1,2,1)




        #fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1)
        ax0.set_title("Old Testset: Slack = 20: Balanced Accuracy = {}".format(bal_acc))
        both_graph = sn.heatmap(old_confusion, xticklabels=beh, yticklabels=beh,
                                annot=True, cmap="YlGnBu", ax=ax0, )
        both_graph.set_xticklabels(both_graph.get_xticklabels(),rotation=50)

        # Gets current figure, sets size (width x height, in inches, given constant dpi)
        #plt.gcf().set_size_inches(12, 12)
        plt.xticks(rotation=60)
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
    annotation_path = "/home/jordan/Desktop/andrew_nih/Annot/pickle_files/old_test/tanner_nih_test_labels/"
    prediction_path = "/home/jordan/Desktop/andrew_nih/Annot/pickle_files/old_test/results_lstm/tanner_nih_test_videos/"
    a = conf_matrix_artemis(config_path, annotation_path, prediction_path)
    a.check_load_csv()
    #a.compute_confusion_matrix()
    a.build_old_new_both_heatmap()
    #a.calculate_metrics()
    #a.build_metrics()

"""
NEXT STEPS:
2. Make 2 more confusion matrices: 1) confusion matrix on old videos, 2) confusion matrix on new videos
3. Output nice graphic that details how many examples you have of each behavior (can use cnfig file for this)
"""
