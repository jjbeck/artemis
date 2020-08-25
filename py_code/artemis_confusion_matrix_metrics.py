import yaml
import glob
import numpy as np
import collections
import pandas as pd
from sklearn.metrics import confusion_matrix

def check_config_file(config_path):
    with open (config_path) as file:
        config_param = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    return config_param['Boot Round'], config_param['Main Path']

class calculate_confusion():

    def __init__(self, main_path, boot_round):
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
        self.prediction_paths = {}
        self.main_path = main_path
        for boot in np.arange(0,boot_round+1):
            self.prediction_paths[main_path + 'csv_b{}/'.format(boot)] = main_path + "csv_b{}/".format(boot)
        self.annotation_path = main_path + "pickle_files/test/"


    def check_load_csv(self):
        # TODO: First iterate through pkl test files. If there is a matching csv file, append both to respective
        #  list. This will guarantee that element at each index in list corresponds to respective element at index in
        #  other list.
        dict_of_predictions = collections.defaultdict(list)
        dict_of_annotations = []
        dict_of_overlap = collections.defaultdict(list)

        try:
            for key in self.prediction_paths:
                for csv in glob.glob(self.prediction_paths[key] + '*.csv'):
                    removed_double_backslash = csv.replace("\\", "/")
                    raw_name = removed_double_backslash.replace(self.prediction_paths[key], '').replace('.csv', '')
                    dict_of_predictions[key].append(raw_name)
        except:
            print('No CSV file in directory. Transfer some and run again')
        # Suffix for rebuilding pickle name.
        pickle_suffix = ''
        try:
            for picklefile in glob.glob(self.annotation_path + '*.p'):
                removed_double_backslash = picklefile.replace("\\", "/")
                raw_pkl_name = removed_double_backslash.replace(self.annotation_path, "").replace("_test.p", "")
                dict_of_annotations.append(raw_pkl_name)
        except:
            print('No Pickle file in directory. Transfer some and run again')

        for key in dict_of_predictions:
            for pred_file in dict_of_predictions[key]:
                if pred_file in dict_of_annotations:
                    dict_of_overlap[key].append(pred_file)
        return dict_of_overlap

    def get_predicted_true_labels(self, dict_of_overlap, main_path):
        pred_data_dict = collections.defaultdict(dict)
        annot_data_dict = collections.defaultdict(dict)

        for boot in dict_of_overlap:
            for prediction_path in dict_of_overlap[boot]:
                pred_full_path = boot + prediction_path + '.csv'
                annot_full_path = main_path + "pickle_files/test/" + prediction_path + '_test.p'
                pred_data = pd.read_csv(pred_full_path, names=['frame', 'pred']).drop_duplicates(subset='frame')
                annot_data = pd.read_pickle(annot_full_path)
                annot_data = annot_data[annot_data['pred'] != 'none']
                annot_data['pred'] = annot_data['pred'].apply(lambda x: self.BEHAVIOR_NAMES.get(x))
                pred_data_dict[boot][prediction_path] = pred_data
                annot_data_dict[prediction_path] = annot_data

        y_pred = collections.defaultdict(list)
        y_true = collections.defaultdict(list)
        y_pred_con = []
        y_true_con = []
        for boot in pred_data_dict:
            for video in pred_data_dict[boot]:
                pred_data_for_annot = pred_data_dict[boot][video].loc[pred_data_dict[boot][video]['frame'].isin(annot_data_dict[video]['frame'])]
                y_pred_con.append(pred_data_for_annot['pred'])
                y_true_con.append(annot_data_dict[video]['pred'])
            y_pred[boot] = pd.concat(y_pred_con)
            y_true[boot] = pd.concat(y_true_con)
            y_pred_con = []
            y_true_con = []

        return y_pred, y_true

    def compute_confusion_matrix(self, y_pred, y_true):
        """
        :param csv: optional argument of list of csvs.
        :param pkl:
        :return:
        """
        y_pred = y_pred
        y_true = y_true

        # Labels array of dimensions (n_classes)
        labels = [mapping[0] for mapping in list(self.BEHAVIOR_LABELS.items()) if mapping[1] != 'none']

        conf_matrix_all = collections.defaultdict(list)
        for boot in y_pred:
            conf_matrix_all[boot] = confusion_matrix(y_pred=y_pred[boot], y_true=y_true[boot], labels=labels,
                                       normalize='true')

            conf_matrix_all[boot] = np.round(conf_matrix_all[boot], decimals=2)

        print(f'Confusion matrix: {conf_matrix_all}')

        return conf_matrix_all

    def return_old_new(self, dict_of_overlap):
        """
        :param dict_of_overlap:
        :return two dictionaries of boot: video_string for old and new videos
        """
        old_dict_of_overlap = collections.defaultdict(list)
        new_dict_of_overlap = collections.defaultdict(list)

        for boot in dict_of_overlap:
            for video in dict_of_overlap[boot]:
                if 'old' in video:
                    old_dict_of_overlap[boot].append(video)
                elif 'new' in video:
                    new_dict_of_overlap[boot].append(video)

        return old_dict_of_overlap, new_dict_of_overlap

