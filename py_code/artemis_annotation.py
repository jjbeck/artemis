import cv2
import pandas as pd
import artemis_annotation_display
import artemis_annotation_calculation
import subprocess
class artemis():

    def __init__(self, main_path, rsync_path, address):
        """
        Initialize variables for script and dictionary with
        keys to behavior label. You can set up this
        dictionary however you like - pickle file with save
        keys to frame, while values will be displayed on video.
        :param main_path:
       :
        """

        self.back = False
        self.forward = False
        self.random = False
        self.main_path = main_path
        self.rsync_path = rsync_path
        self.address = address
        self.BEHAVIOR_LABELS = {
            0: "drink",
            2: "groom",
            1: "eat",
            3: "hang",
            4: "sniff",
            5: "rear",
            6: "rest",
            7: "walk",
            8: "eathand",
            9: "none"
        }
        self.pick_keys = {
            48: 0,
            49: 1,
            50: 2,
            51: 3,
            52: 4,
            53: 5,
            54: 6,
            55: 7,
            56: 8,
            57: 9,
        }
        # empty dataframe to store annotations for session
        self.annot_data = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
        self.annotating = True

    def organize_files(self):
        #ls = subprocess.check_output(
            #"ssh {} ls {}".format(self.address, self.main_path) + "/videos_not_done/", shell=True)
        ls = subprocess.check_output(
         "ls {}".format(self.main_path) + "/videos_not_done/", shell=True)

        video_files = ls.decode('utf-8').split('\n')
        if len(video_files) == 1:
            print("No videos in directory. Add videos and re-run artemis_annotation")

        a = artemis_annotation_display.display()
        video_file, video_rsync, test_or_train = a.pick_file(video_files, self.main_path, self.rsync_path)

        #subprocess.run("rsync {}:{} {}".format(self.address, self.main_path + "/config.yaml",
                                                                            #self.rsync_path + "/config_yaml"),
                       #shell=True)
        a = artemis_annotation_calculation.metrics()
        boot_round = a.calculate_config(self.rsync_path + '/config.yaml')

        if test_or_train == "test":
            pickle_path = self.main_path + "/pickle_files/test" + video_file[video_file.rfind('/'):video_file.rfind(".")] + "_test.p"
            pickle_rsync = self.rsync_path + "/pickle_files/test" + video_file[video_file.rfind('/'):video_file.rfind(".")] + "_test.p"
        else:
            pickle_path = self.main_path + "/pickle_files/train" + video_file[video_file.rfind('/'):video_file.rfind(".")] + "_boot{}.p".format(boot_round)
            pickle_rsync = self.rsync_path + "/pickle_files/train" + video_file[video_file.rfind('/'):video_file.rfind(".")] + "_boot{}.p".format(boot_round)

    def create_file_name(self,base_path, extension_path, file_name, video=None, csv=None, pickle=None):
