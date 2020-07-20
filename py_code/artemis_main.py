import artemis_annotation
class artemis:

    def __init__(self):
        print("artemis loaded")

    def annotate(self):
        a = artemis_annotation.artemis('/home/jordan/Desktop/Annot', interval=30)
        video_path, pickle_path, pickle_rsync_path, csv_path, csv_rsync_path = a.organize_files()
        a.get_usable_dataframe(video_path, pickle_path, csv_path, pickle_rsync_path = pickle_rsync_path, csv_rsync_path = csv_rsync_path)
        a.load_video(video_path)
        print(video_path)
        print(pickle_path)
        print(pickle_rsync_path)
        print(csv_path)
        print(csv_rsync_path)


    #def record_video(self):

    #def compute_confusion_matrix(self):

    #def bootstrap(self):

    #def run_inference(self):


a = artemis()
a.annotate()

