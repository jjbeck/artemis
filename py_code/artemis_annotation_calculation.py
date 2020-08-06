import os
import subprocess

import chardet
import pandas as pd
import pims
import yaml


def calculate_frames(video_path):
    """
    Calculates total number of frames in video at video path.
    :param video_path: Path to video.
    :return: Integer of total number of frames.
    """
    video = pims.PyAVReaderTimed(video_path)
    number_of_frames = len(video)
    return number_of_frames


def detect_encoding(path):
    """
    Detects encoding of file.
    :param path: path to file
    :return: string of encoding
    """
    with open(path, 'rb') as f:
        result = chardet.detect(f.read(1024 ** 2))
        return result


def mask_keycode(usr_in):
    """
    Handles masking keycodes. This is here because waitKey does not work for left and right arrow keys on windows.
    It is replaced by waitKeyEx, which returns different keycodes for the left and right arrow keys on Windows than
    other OSes. This handles those cases.
    :param usr_in: Key code input by user.
    :return: Key code that artemis uses.
    """
    masked = usr_in & 0xFF
    # Left arrow key
    if usr_in == 2424832:
        masked = ord('Q')
    # Right arrow key
    if usr_in == 2555904:
        masked = ord('S')
    # Esc button (windows)
    if usr_in == 27:
        masked = ord('\x1b')

    return masked


class metrics:

    def __init__(self):
        self.csv_path = None
        self.csv_rsync_path = None

    def set_csv_path(self, csv_path):
        self.csv_path = csv_path

    def create_config(self, config_path):
        default_boot_round = 1
        data = [{"Boot Round": default_boot_round}]
        with open(config_path, 'w') as new_config:
            yaml.dump(data, new_config)
        return default_boot_round

    def calculate_config(self, config_path):
        try:
            with open(config_path) as file:
                config_param = yaml.load(file, Loader=yaml.FullLoader)
                file.close()
        except FileNotFoundError:
            print(f"No config file found, creating one now at {config_path}.")
            boot = self.create_config(config_path)
            # Default boot round when no config path is found.
            return boot
        # Iterates through yaml data, and gets value for first occurrence of Boot Round
        return config_param["Boot Round"]

    def determine_folder_hierarchy(self, main_path, rsync_path=None):
        if rsync_path is not None:
            print("checking server hiearchy")

        for root, dirs, files in os.walk(main_path):
            if 'videos_not_done' in dirs and 'pickle_files' in dirs and 'csv_not_done' in dirs:
                if os.path.isdir(main_path + '/pickle_files/test') == True and os.path.isdir(
                        main_path + '/pickle_files/train') == True:
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
        print(f'Video file: {video_file}')
        # video_path = main_path + '/videos_not_done/' + video_file
        video_path = video_file
        # csv_path = main_path + '/csv_not_done/' + video_file[video_file.rfind('/'):video_file.rfind(".")] + ".csv"
        csv_path = main_path + '/csv_not_done' + video_file[video_file.rfind('/'):video_file.rfind(".")] + ".csv"

        if test_or_train == "test":
            pickle_path = main_path + "/pickle_files/test" + video_file[
                                                             video_file.rfind('/'):video_file.rfind(".")] + "_test.p"
            if rsync_path is not None:
                pickle_rsync_path = rsync_path + "/pickle_files/test" + video_file[
                                                                        video_file.rfind('/'):video_file.rfind(
                                                                            ".")] + "_test.p"
                csv_rsync_path = rsync_path + '/csv_not_done/' + video_file[
                                                                 video_file.rfind('/'):video_file.rfind(".")] + ".csv"
            else:
                pickle_rsync_path = None
                csv_rsync_path = None
        else:
            pickle_path = main_path + "/pickle_files/train" + video_file[video_file.rfind('/'):video_file.rfind(
                ".")] + "_boot{}.p".format(boot_round)
            if rsync_path is not None:
                pickle_rsync_path = rsync_path + "/pickle_files/train" + video_file[
                                                                         video_file.rfind('/'):video_file.rfind(
                                                                             ".")] + "_boot{}.p".format(boot_round)
                csv_rsync_path = rsync_path + '/csv_not_done/' + video_file[
                                                                 video_file.rfind('/'):video_file.rfind(".")] + ".csv"
            else:
                pickle_rsync_path = None
                csv_rsync_path = None
        self.csv_path = csv_path
        self.csv_rsync_path = csv_rsync_path
        return video_path, pickle_path, pickle_rsync_path, csv_path, csv_rsync_path

    def get_prediction(self, frame_header, interval=None):
        """
        :param frame_header: Frame to start at
        :param interval: Interval size. None assumes interval of 1
        :return: String of label that is most common across interval, or label at frame header if interval arg is not
        filled.
        """
        df = pd.read_csv(self.csv_path)
        if interval is not None:
            return df.iloc[frame_header:(frame_header + interval)]['label'].mode()
        else:
            return df.iloc[frame_header:(frame_header + interval)]['label']
