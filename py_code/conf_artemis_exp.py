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

    def __init__(self, test_path):
        self.test_path = test_path

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
        self.analyze_leo = []
        self.analyze_sami = []
        self.analyze_jordan = []

    def check_load_csv(self):
        # TODO: First iterate through pkl test files. If there is a matching csv file, append both to respectile
        #  list. This will guarantee that element at each index in list corresponds to respective element at index in
        #  other list.
        set_of_sami = set()
        set_of_leo = set()
        set_of_jordan = set()
        try:
            for leo in glob.glob(self.test_path + '*.p'):
                # We clean the csv name to find just the file name:
                #   csv_not_done/abc.csv --> abc
                if "leo" in leo:
                    file_name = leo.replace(self.test_path, '').replace('_leo.p', '')
                    set_of_leo.add(file_name)
            for sami in glob.glob(self.test_path + '*.p'):
                # We clean the csv name to find just the file name:
                #   csv_not_done/abc.csv --> abc
                if "sami" in sami:
                    file_name = sami.replace(self.test_path, '').replace('_sami.p', '')
                    set_of_sami.add(file_name)
            for jordan in glob.glob(self.test_path + '*.p'):
                # We clean the csv name to find just the file name:
                #   csv_not_done/abc.csv --> abc
                if "sami" not in jordan and "leo" not in jordan:
                    file_name = jordan.replace(self.test_path, '').replace('.p', '')
                    set_of_jordan.add(file_name)




        except:
            print('No CSV file in directory. Transfer some and run again')



        # Suffix for rebuilding pickle name.
        pickle_suffix = ''
        common_files = list(set_of_jordan.intersection(set_of_sami))
        #all_files = list(set_of_jordan.intersection(common_files))
        # We rebuild list of csv and pickles from this intersection.
        for file in common_files:
            sami_name_rebuilt = self.test_path + file + '_sami.p'
            self.analyze_sami.append(sami_name_rebuilt)
            #leo_name_rebuilt = self.test_path + file + '_leo.p'
            #self.analyze_leo.append(leo_name_rebuilt)
            jordan_name_rebuilt = self.test_path + file + '.p'
            self.analyze_jordan.append(jordan_name_rebuilt)

    def get_predicted_true_labels(self, sami_data, jordan_data):
    #def get_predicted_true_labels(self, leo_data, sami_data, jordan_data):
        #leo_data_df = []
        sami_data_df = []
        jordan_data_df = []
        """
        for leo in leo_data:
            data = pd.read_pickle(leo)
            data = data[data['pred'] != 'none']
            data['pred'] = data['pred'].apply(lambda x: self.BEHAVIOR_NAMES.get(x))
            leo_data_df.append(data)
        """
        for sami in sami_data:
            data = pd.read_pickle(sami)
            data = data[data['pred'] != 'none']
            data['pred'] = data['pred'].apply(lambda x: self.BEHAVIOR_NAMES.get(x))
            sami_data_df.append(data)
        for jordan in jordan_data:
            data = pd.read_pickle(jordan)
            data = data[data['pred'] != 'none']
            data['pred'] = data['pred'].apply(lambda x: self.BEHAVIOR_NAMES.get(x))
            jordan_data_df.append(data)

        y_jordan_sami = []
        y_sami = []
        #y_leo = []
        #y_jordan_leo = []
        for sami, jordan in zip(jordan_data_df, sami_data_df):
            sami_data_for_jordan = sami.loc[sami['frame'].isin(jordan['frame'])]
            jordan_data_in_sami = jordan.loc[jordan['frame'].isin(sami["frame"])]
            y_sami.append(sami_data_for_jordan['pred'])
            y_jordan_sami.append(jordan_data_in_sami['pred'])

        """
        for leo, jordan in zip(jordan_data_df, leo_data_df):
            leo_data_for_jordan = leo.loc[leo['frame'].isin(jordan['frame'])]
            y_leo.append(leo_data_for_jordan['pred'])
            y_jordan_leo.append(jordan['pred'])
        """


        #y_leo = pd.concat(y_leo)
        y_sami = pd.concat(y_sami)
        #y_jordan_leo = pd.concat(y_jordan_leo)
        y_jordan_sami = pd.concat(y_jordan_sami)
        print(len(y_sami))
        print(len(y_jordan_sami))
        return y_sami,y_jordan_sami
        #return y_leo, y_sami, y_jordan_leo, y_jordan_sami

    def compute_confusion_matrix(self, sami=None, jordan=None):
    #def compute_confusion_matrix(self, leo=None, sami=None, jordan=None):
        """
        :param csv: optional argument of list of csvs.
        :param pkl:
        :return:
        """
        #leo_data = leo
        sami_data = sami
        jordan_data= jordan
        if sami_data is None:
            #leo_data = self.analyze_leo
            sami_data = self.analyze_sami
            jordan_data = self.analyze_jordan

        # Labels array of dimensions (n_classes)
        labels = [mapping[0] for mapping in list(self.BEHAVIOR_LABELS.items()) if mapping[1] != 'none']

        #y_leo, y_sami, y_jordan_leo, y_jordan_sami = self.get_predicted_true_labels(leo_data, sami_data, jordan_data)
        y_sami, y_jordan_sami = self.get_predicted_true_labels(sami_data, jordan_data)

        #conf_matrix_leo = confusion_matrix(y_pred=y_leo, y_true=y_jordan_leo, labels=labels,
        #                               normalize='true')
        conf_matrix_sami = confusion_matrix(y_pred=y_sami, y_true=y_jordan_sami, labels=labels,
                                           normalize='true')

        #conf_matrix_leo = np.round(conf_matrix_leo, decimals=2)

        conf_matrix_sami = np.round(conf_matrix_sami, decimals=2)



        return conf_matrix_sami

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
            #leo_files = self.analyze_leo
            sami_files = self.analyze_sami
            jordan_files = self.analyze_jordan

        # TODO: Currently, each csv/pkl file in function arguments must contain self.csv_path or self.pickle_path
        #  respectively for this function to work. This is because we first clean up the strings by removing the
        #  file path prefix. This needs to be changed so that we split on some character that delineates the end
        #  of the path prefix and the start of the actual file name.

        #leo_data = [pkl.replace(self.test_path, '') for pkl in leo_files]
        sami_data = [pkl.replace(self.test_path, '') for pkl in sami_files]
        jordan_data = [pkl.replace(self.test_path, '') for pkl in jordan_files]




        #leo_csv = [self.test_path + csv for csv in leo_data]
        sami_csv = [self.test_path + pkl for pkl in sami_data]
        jordan_csv = [self.test_path + pkl for pkl in jordan_data]
        return sami_csv, jordan_csv


    def build_old_new_both_heatmap(self):
        sami_csv, jordan_csv = self.return_old_new()

        sami_conf = self.compute_confusion_matrix()


        beh = ['drink', 'eat', 'groom',
               'hang', 'sniff', 'rear', 'rest',
               'walk', 'eathand']

        fig = plt.figure(figsize=(12,12))


        ax0 = fig.add_subplot(1,2,1)
        ax1 = fig.add_subplot(1,2,2)

        fig.subplots_adjust(hspace=0.35)




        ax0.set_title("Leo and Jordan Accuracy")
        #both_graph = sn.heatmap(leo_conf, xticklabels=beh, yticklabels=beh,
         #                       annot=True, cmap="YlGnBu", ax=ax0, )
        #both_graph.set_xticklabels(both_graph.get_xticklabels(),rotation=50)
        #both_graph.set_yticklabels(both_graph.get_yticklabels(), rotation=60)
        ax1.set_title("Sami and Jordan Accuracy")
        old_graph = sn.heatmap(sami_conf, xticklabels=beh, yticklabels=beh, annot=True, cmap="YlGnBu", ax=ax1)
        old_graph.set_xticklabels(old_graph.get_xticklabels(), rotation=50)
        old_graph.set_yticklabels(old_graph.get_yticklabels(), rotation=60)

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
    test_path = "/home/jordan/Desktop/andrew_nih/Annot/pickle_files/test/"
    a = conf_matrix_artemis(test_path)
    a.check_load_csv()
    a.compute_confusion_matrix()
    a.build_old_new_both_heatmap()

"""
NEXT STEPS:
2. Make 2 more confusion matrices: 1) confusion matrix on old videos, 2) confusion matrix on new videos
3. Output nice graphic that details how many examples you have of each behavior (can use cnfig file for this)
"""
