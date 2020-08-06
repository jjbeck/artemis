import artemis_confusion_matrix_metrics
class confusion_matrix():

    def __init__(self, config_path):
        self.boot_round = artemis_confusion_matrix_metrics.check_config_file(config_path)
        print(self.boot_round)