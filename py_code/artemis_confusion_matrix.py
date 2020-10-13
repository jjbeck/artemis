import artemis_confusion_matrix_metrics
import artemis_confusion_matrix_display


class confusion_matrix:

    def __init__(self, config_path):
        self.boot_round, self.main_path, self.pred_file_path, self.fig_title = artemis_confusion_matrix_metrics.check_file_path(config_path)

    def organize_files(self):
        conf_met = artemis_confusion_matrix_metrics.calculate_confusion(self.main_path, self.pred_file_path)
        # return dictionary with {['Boot round']:[overlapping videos in boot round and test set]
        dict_of_overlap = conf_met.check_load_csv()
        y_true, y_pred = conf_met.get_predicted_true_labels()
        conf_matrix = conf_met.compute_confusion_matrix(y_pred, y_true)



        artemis_confusion_matrix_display.build_heatmap(conf_matrix, self.boot_round, self.fig_title)

