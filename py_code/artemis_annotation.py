import cv2
import pandas as pd
import artemis_annotation_display
import artemis_annotation_calculation
import subprocess
import sys
import pims
class artemis():


    def __init__(self, main_path, rsync_path=None, address=None, interval=30, playback_speed=1):
        """
        Initialize variables for script and dictionary with
        keys to behavior label. You can set up this
        dictionary however you like - pickle file with save
        keys to frame, while values will be displayed on video.
        :param main_path:
       :
        """
        #set state for going back forward and choosing random frame
        self.back = False
        self.forward = False
        self.random = False
        self.annotating = True
        #Set main path to files
        self.main_path = main_path
        #Set path to local computer if there is one
        self.rsync_path = rsync_path
        #If rsyncing, this sets username and address
        self.address = address
        #set other attributes to run artemis
        self.interval = interval
        self.playback_speed = playback_speed
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


    def organize_files(self):
        a = artemis_annotation_display.display()
        b = artemis_annotation_calculation.metrics()
        file_org = b.determine_folder_hierarchy(self.main_path, rsync_path=self.rsync_path)
        if file_org == False:
            sys.exit('Error: Directories not organized properly. See documentation for further help')

        video_files = b.check_video_files(rsync_path=self.rsync_path)
        if video_files is not None:
            video_file, video_rsync, test_or_train = a.pick_file(video_files, self.main_path, self.rsync_path)
            subprocess.run("rsync {}:{} {}".format(self.address, self.main_path + "/config.yaml",
                                                                                self.rsync_path + "/config_yaml"),
                           shell=True)
            boot_round = a.calculate_config(self.rsync_path + '/config.yaml')
            if len(video_files) == 1:
                print("No videos in directory. Add videos and re-run artemis_annotation")
        else:
            video_file, test_or_train = a.choose_local_file(self.main_path + "/videos_not_done/")
            boot_round = b.calculate_config(self.main_path + '/config.yaml')

        video_path, pickle_path, pickle_rsync_path, csv_path, csv_rsync_path = b.create_file_names(video_file, self.main_path, test_or_train, boot_round, rsync_path=self.rsync_path)

        return video_path, pickle_path, pickle_rsync_path, csv_path, csv_rsync_path

    #def get_usable_dataframe(self):

    #load_video(self, video_file):
        #self.cap = pims.PyAVReaderTimed(video_file)

    #def annotate_video
        #get prediction
        #display interval with prediction
        #get key press and save into dataframe
        #when done save to pickle file

    #def save_pickle



    #done_with_video



"""
1. Add way to check for prescence of configd.yaml file
"""