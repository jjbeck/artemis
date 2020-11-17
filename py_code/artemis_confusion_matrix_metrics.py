import yaml
import glob
import numpy as np
import collections
import pandas as pd
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import simpledialog

def check_file_path(config_path):
    root = tk.Tk()
    root.withdraw()
    pred_file_name = askdirectory(initialdir="~/Desktop/andrew_nih/Annot/",
                                  title="Select Predictions Path")
    fig_title = simpledialog.askstring(title="Figure Title",
                                                prompt="Enter figure title")

    with open (config_path) as file:
        config_param = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    return config_param['Boot Round'], config_param['Main Path'], pred_file_name, fig_title

class calculate_confusion():

    def __init__(self, main_path, pred_path):
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
            "eathand": 8,
            "none": 9,
        }
        #create empty list to store common ground truth and prediction files
        self.analyze_csv = []
        self.analyze_pickle = []
        self.main_path = main_path
        self.prediction_path = pred_path + '/'
        self.annotation_path = main_path + "pickle_files/test/"

        self.conf_matrix = np.zeros((9,9))


    def check_load_csv(self):
        # TODO: First iterate through pkl test files. If there is a matching csv file, append both to respective
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

        try:
            for picklefile in glob.glob(self.annotation_path + '*.p'):
                file_name = picklefile.replace(self.annotation_path, '').replace('_test.p','')
                set_of_pkl.add(file_name)
        except:
            print('No Pickle file in directory. Transfer some and run again')

        common_files = list(set_of_pkl.intersection(set_of_csv))

        # We rebuild list of csv and pickles from this intersection.
        for file in common_files:
            csv_name_rebuilt = self.prediction_path + file + '.csv'
            self.analyze_csv.append(csv_name_rebuilt)
            pickle_name_rebuilt = self.annotation_path + file + '_test.p'

            self.analyze_pickle.append(pickle_name_rebuilt)

    def get_predicted_true_labels(self):
        csv_data_df = []
        for csv in self.analyze_csv:
            data = pd.read_csv(csv, names=['frame', 'pred']).drop_duplicates(subset='frame')
            csv_data_df.append(data)

        # Dataframe of  all csv data.
        pkl_data_df = []
        for pickle in self.analyze_pickle:
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

        y_pred = pd.concat(y_pred).reset_index(drop=True)
        y_true = pd.concat(y_true).reset_index(drop=True)

        return y_pred, y_true

    def compute_confusion_matrix(self,y_pred,y_true):
        """
        :param csv: optional argument of list of csvs.
        :param pkl:
        :return:
        """
        slack = 20
        for idx,l in y_true.iteritems():
            if l in y_pred[int(max(0, idx - slack / 2)): int(min(idx + 1 + slack / 2, len(y_true)))]:
                self.conf_matrix[l, l] += 1
            else:
                self.conf_matrix[l, y_pred[idx]] += 1

        print(self.conf_matrix)
        row_sums = np.sum(self.conf_matrix, axis=1)
        self.conf_matrix = self.conf_matrix / row_sums[:, np.newaxis]
        return self.conf_matrix

    def slackify(y_true, y_pred, slack=20):
        y_pred_slack = []
        for i, yp in enumerate(y_pred):
            if yp in y_true[int(max(0, i - slack / 2)): int(min(i + 1 + slack / 2, len(y_true)))]:
                y_pred_slack.append(y_true[i])
            else:
                y_pred_slack.append(yp)

        return y_pred_slack



