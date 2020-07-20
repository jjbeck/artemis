from typing import List, Union

import yaml
import subprocess
import os

class metrics():

    def __init__(self):
        print("metrics")

    def calculate_config(self, config_path):
        with open(config_path) as file:
            config_param = yaml.load(file, Loader=yaml.FullLoader)
            file.close()
        return config_param["Boot Round"]

    def determine_folder_hierarchy(self, main_path, rsync_path=None):
        if rsync_path is not None:
            print("checking server hiearchy")

        for root, dirs, files in os.walk(main_path):
            if 'videos_not_done' in dirs and 'pickle_files' in dirs and 'csv_not_done' in dirs:
                if os.path.isdir(main_path + '/pickle_files/test') == True and os.path.isdir(main_path + '/pickle_files/train') == True:
                    return True
        return False

    def check_video_files(self, rsync_path=None):
        if rsync_path is not None:
            ls = subprocess.check_output(
                "ssh {} ls {}".format(self.address, self.main_path) + "/videos_not_done/", shell=True)
            video_files = ls.decode('utf-8').split('\n')
            return video_files
        else:
            return None

    def create_file_names(self, video_file, main_path, test_or_train, boot_round, rsync_path=None):
        video_path = main_path + '/videos_not_done/' + video_file
        csv_path = main_path + '/csv_not_done/' + video_file[video_file.rfind('/'):video_file.rfind(".")] + ".csv"
        if test_or_train == "test":
            pickle_path = main_path + "/pickle_files/test" + video_file[video_file.rfind('/'):video_file.rfind(".")] + "_test.p"
            if rsync_path is not None:
                pickle_rsync_path = rsync_path + "/pickle_files/test" + video_file[video_file.rfind('/'):video_file.rfind(".")] + "_test.p"
                csv_rsync_path = rsync_path + '/csv_not_done/' + video_file[
                                                          video_file.rfind('/'):video_file.rfind(".")] + ".csv"
            else:
                pickle_rsync_path = None
                csv_rsync_path = None
        else:
            pickle_path = main_path + "/pickle_files/train" + video_file[video_file.rfind('/'):video_file.rfind(".")] + "_boot{}.p".format(boot_round)
            if rsync_path is not None:
                pickle_rsync_path = rsync_path + "/pickle_files/train" + video_file[video_file.rfind('/'):video_file.rfind(".")] + "_boot{}.p".format(boot_round)
                csv_rsync_path = rsync_path + '/csv_not_done/' + video_file[
                                                                 video_file.rfind('/'):video_file.rfind(".")] + ".csv"
            else:
                pickle_rsync_path = None
                csv_rsync_path = None
        return video_path, pickle_path, pickle_rsync_path, csv_path, csv_rsync_path


