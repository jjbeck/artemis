import artemis_confusion_matrix_metrics
import artemis_confusion_matrix_display
class confusion_matrix():

    def __init__(self, config_path):
        self.boot_round, self.main_path = artemis_confusion_matrix_metrics.check_config_file(config_path)

    def organize_files(self):
        conf_met = artemis_confusion_matrix_metrics.calculate_confusion(self.main_path, self.boot_round)
        #return dictionary with {['Boot round']:[overlapping videos in boot round and test set]
        dict_of_overlap = conf_met.check_load_csv()
        old_dict_of_overlap, new_dict_of_overlap = conf_met.return_old_new(dict_of_overlap)
        y_true_old, y_pred_old = conf_met.get_predicted_true_labels(old_dict_of_overlap, self.main_path)
        y_true_new, y_pred_new = conf_met.get_predicted_true_labels(new_dict_of_overlap, self.main_path)
        y_true_both, y_pred_both = conf_met.get_predicted_true_labels(dict_of_overlap, self.main_path)
        conf_matrix_old = conf_met.compute_confusion_matrix(y_pred_old, y_true_old)
        conf_matrix_new = conf_met.compute_confusion_matrix(y_pred_new, y_true_new)
        conf_matrix_both = conf_met.compute_confusion_matrix(y_pred_both, y_true_both)

        artemis_confusion_matrix_display.build_heatmap(conf_matrix_both, conf_matrix_old, conf_matrix_new)



