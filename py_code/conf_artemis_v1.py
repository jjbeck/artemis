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

    def get_predicted_true_labels(self, csv_data, pickle_data):
        csv_data_df = []
        for csv in csv_data:
            data = pd.read_csv(csv, names=['frame', 'pred']).drop_duplicates(subset='frame')
            csv_data_df.append(data)
        # Dataframe of  all csv data.
        pkl_data_df = []
        for pickle in pickle_data:
            data = pd.read_pickle(pickle)
            data = data[data['pred'] != 'none']
            data['pred'] = data['pred'].apply(lambda x: self.BEHAVIOR_NAMES.get(x))
            pkl_data_df.append(data)
        y_pred = []
        y_true = []
        for csv_, pkl in zip(csv_data_df, pkl_data_df):
            csv_data_for_pkl = csv_.loc[csv_['frame'].isin(pkl['frame'])]
            y_pred.append(csv_data_for_pkl['pred'])
            y_true.append(pkl['pred'])

        y_pred = pd.concat(y_pred)
        y_true = pd.concat(y_true)
        return y_pred, y_true

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

        y_pred, y_true = self.get_predicted_true_labels(csv_data, pkl_data)

        conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels,
                                       normalize='true')

        conf_matrix = np.round(conf_matrix,decimals=2)

        if csv is None or pkl is None:
            self.confusion_matrix = conf_matrix

        return conf_matrix

    def return_old_new(self, version, csv=None, pkl=None):
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

        if version == 'old':
            old_csv = [self.prediction_path + csv for csv in csv_data if csv.rsplit("_")[2] == 'old']
            old_pkl = [self.annotation_path + pkl for pkl in pkl_data if pkl.rsplit("_")[2] == 'old']
            return old_csv, old_pkl
        if version == 'new':
            new_csv = [self.prediction_path + csv for csv in csv_data if csv.rsplit("_")[2] == 'new']
            new_pkl = [self.annotation_path + pkl for pkl in pkl_data if pkl.rsplit("_")[2] == 'new']
            return new_csv, new_pkl

        raise Exception('Error: version must be \'old\' or \'new\'.')

    def build_old_new_both_heatmap(self):
        old_csv, old_pkl = self.return_old_new(version='old')
        new_csv, new_pkl = self.return_old_new(version='new')

        old_confusion = self.compute_confusion_matrix(csv=old_csv, pkl=old_pkl)
        new_confusion = self.compute_confusion_matrix(csv=new_csv, pkl=new_pkl)
        both_confusion = self.compute_confusion_matrix()

        bout, sample = self.calculate_metrics()

        beh = ['drink', 'eat', 'groom',
               'hang', 'sniff', 'rear', 'rest',
               'walk', 'eathand']

        fig = plt.figure(figsize=(12,12))


        ax0 = fig.add_subplot(3,2,1)
        ax1 = fig.add_subplot(3,2,3)
        ax2 = fig.add_subplot(3,2,5)
        bt = fig.add_subplot(2,2,2)
        smp = fig.add_subplot(2,2,4)
        fig.subplots_adjust(hspace=0.35)



        #fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1)
        ax0.set_title("Both")
        both_graph = sn.heatmap(both_confusion, xticklabels=beh, yticklabels=beh,
                                annot=True, cmap="YlGnBu", ax=ax0, )
        both_graph.set_xticklabels(both_graph.get_xticklabels(),rotation=50)
        ax1.set_title("Old")
        old_graph = sn.heatmap(old_confusion, xticklabels=beh, yticklabels=beh, annot=True, cmap="YlGnBu", ax=ax1)
        old_graph.set_xticklabels(old_graph.get_xticklabels(), rotation=50)
        ax2.set_title("New")
        new_graph = sn.heatmap(new_confusion, xticklabels=beh, yticklabels=beh, annot=True, cmap="YlGnBu", ax=ax2)
        new_graph.set_xticklabels(new_graph.get_xticklabels(), rotation=50)
        a = bout.plot.bar(ax=bt, title="Bouts for each behavior", legend=False, yticks=np.linspace(0,150,11))
        a.set_xticklabels(a.get_xticklabels(), rotation=50)
        b = sample.plot.bar(ax=smp, title="Samples(frames) for each behavior",
                                 yticks=np.linspace(0, 20000, 11))
        b.set_xticklabels(b.get_xticklabels(), rotation=50)
        # Gets current figure, sets size (width x height, in inches, given constant dpi)
        #plt.gcf().set_size_inches(12, 12)
        plt.xticks(rotation=60)
        plt.show()

    def unpack_metrics(self, drink=[0, 0], eat=[0, 0], groom=[0, 0],
                       hang=[0, 0], sniff=[0, 0], rear=[0, 0], rest=[0, 0], walk=[0, 0], eathand=[0, 0], none=[0, 0]):
        return {"drink": drink, "eat": eat, "groom": groom, "hang": hang,
                "sniff": sniff, "rear": rear, "rest": rest, "walk": walk, "eathand": eathand}

    def calculate_metrics(self):
        # create empyty dictionaries to append bouts and samples to
        new_sum = {"drink": [0, 0],
                        "eat": [0, 0],
                        "groom": [0, 0],
                        "hang": [0, 0],
                        "sniff": [0, 0],
                        "rear": [0, 0],
                        "rest": [0, 0],
                        "walk": [0, 0],
                        "eathand": [0, 0]}
        old_sum = {"drink": [0, 0],
                        "eat": [0, 0],
                        "groom": [0, 0],
                        "hang": [0, 0],
                        "sniff": [0, 0],
                        "rear": [0, 0],
                        "rest": [0, 0],
                        "walk": [0, 0],
                        "eathand": [0, 0]}

        bout = pd.DataFrame()
        sample = pd.DataFrame()

        for key in self.new_test_set_beh.keys():
            a = self.unpack_metrics(**self.new_test_set_beh[key])
            for beh in a.keys():
                new_sum[beh] = list(map(add, new_sum[beh],a[beh]))

        for key in self.old_test_set_beh.keys():
            a = self.unpack_metrics(**self.old_test_set_beh[key])
            for beh in a.keys():
                old_sum[beh] = list(map(add, old_sum[beh],a[beh]))

        new_sum = pd.DataFrame.from_dict(new_sum,orient="index")
        old_sum = pd.DataFrame.from_dict(old_sum,orient="index")
        bout["New Video"] = new_sum[0]
        bout["Old Video"] = old_sum[0]
        sample["New Video"] = new_sum[1]
        sample["Old Video"] = old_sum[1]

        return bout, sample



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
    a.check_load_csv()
    a.compute_confusion_matrix()
    a.build_old_new_both_heatmap()
    #a.calculate_metrics()
    #a.build_metrics()

"""
NEXT STEPS:
2. Make 2 more confusion matrices: 1) confusion matrix on old videos, 2) confusion matrix on new videos
3. Output nice graphic that details how many examples you have of each behavior (can use cnfig file for this)
"""
