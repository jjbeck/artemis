import artemis_annotation
import artemis_confusion_matrix


class artemis:

    def __init__(self):
        print("artemis loaded")

    def annotate(self, annotation_path):
        a = artemis_annotation.artemis(annotation_path, interval=20, encoding='iso-8859-1')
        video_path, pickle_path, pickle_rsync_path, csv_path, csv_rsync_path = a.organize_files()
        final_pickle_path = pickle_path
        final_csv_path = csv_path
        if pickle_rsync_path is not None:
            final_pickle_path = pickle_rsync_path

        if csv_rsync_path is not None:
            final_csv_path = csv_rsync_path

        # Load data -> Usable dataframe -> Annotate Video always.
        a.load_data(video_path, pickle_path=final_pickle_path, csv_path=final_csv_path)
        usable_df = a.get_usable_dataframe(video_path, final_pickle_path, final_csv_path)
        a.annotate_video(usable_df, pickle_path=final_pickle_path, predictions_csv=csv_path)


    #def record_video(self):

    def compute_confusion_matrix(self, config_path):
        conf = artemis_confusion_matrix.confusion_matrix(config_path)
        conf.organize_files()

    #def bootstrap(self):

    #def run_inference(self):


a = artemis()
#a.annotate('/home/jordan/Desktop/andrew_nih/Annot')
a.compute_confusion_matrix('/home/jordan/Desktop/andrew_nih/Annot/config.yaml')

"""
Note: CSV Files have the us-ascii charset.
"""