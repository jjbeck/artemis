import artemis_annotation
class artemis:

    def __init__(self):
        print("artemis loaded")

    def annotate(self):
        a = artemis_annotation.artemis('/home/jordan/Desktop/Annot', '/home/jordan/Desktop/Annot', None)
        a.organize_files()


    #def record_video(self):

    #def compute_confusion_matrix(self):

    #def bootstrap(self):

    #def run_inference(self):


a = artemis()
a.annotate()

