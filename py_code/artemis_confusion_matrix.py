import artemis_confusion_matrix_metrics
class confusion_matrix():

    def __init__(self, config_path):
        self.boot_round, self.main_path = artemis_confusion_matrix_metrics.check_config_file(config_path)

    def organize_files(self):
        conf_met = artemis_confusion_matrix_metrics.calculate_confusion(self.main_path, self.boot_round)
        #return dictionary with {['Boot round']:[overlapping videos in boot round and test set]
        dict_of_overlap = conf_met.check_load_csv()
        print(dict_of_overlap)
        y_true, y_pred = conf_met.get_predicted_true_labels(dict_of_overlap, self.main_path)
        conf_matrix_all = conf_met.compute_confusion_matrix(y_pred,y_true)



