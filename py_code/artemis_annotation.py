import pickle

import cv2
import pandas as pd
import artemis_annotation_display
import artemis_annotation_calculation
import subprocess
import sys
import os
import pims
import numpy as np
import traceback


class artemis:

    def __init__(self, main_path, rsync_path=None, address=None, interval=30, playback_speed=1, encoding=None):
        """
        Initialize variables for script and dictionary with
        keys to behavior label. You can set up this
        dictionary however you like - pickle file with save
        keys to frame, while values will be displayed on video.
        :param main_path: Path to videos
        :param rsync_path: If doing over rsync, rsync path.
        :param address
        :param interval: frame interval
        :param encoding: encoding to csv files
        """
        # Field variables containing display and metrics objects
        self.cap = None
        self.display = artemis_annotation_display.display()
        self.metrics = artemis_annotation_calculation.metrics()
        self.encoding = encoding
        self.frame_header = None
        self.dataset_test_train = None
        self.csv_path = None
        self.csv_df = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
        # Pickle dataframe that gets reset after every session. Is used to save into pickle files.
        self.pickle_data = pd.DataFrame(columns=['frame', 'label'], dtype='int64')
        self.pickle_cache = pd.DataFrame(columns=['frame', 'label'], dtype='int64')
        # Set to length of frames that have not yet been labelled.
        self.frames_labelled_in_session = 0
        # Set main path to files
        self.main_path = main_path
        # Set path to local computer if there is one
        self.rsync_path = rsync_path
        # If rsyncing, this sets username and address
        self.address = address
        # set other attributes to run artemis
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
        self.handle_input = {
            # utility fn
            ord('y'): self.letter,
            ord(' '): self.letter,
            ord('Q'): self.letter,
            ord('S'): self.letter,
            ord('r'): self.letter,
            ord('s'): self.letter,
            ord('p'): self.letter,
            ord('\x1b'): self.letter,
            # 1 big annotation function
            ord('0'): self.label,
            ord('1'): self.label,
            ord('2'): self.label,
            ord('3'): self.label,
            ord('4'): self.label,
            ord('5'): self.label,
            ord('6'): self.label,
            ord('7'): self.label,
            ord('8'): self.label,
            ord('9'): self.label
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

    def letter(self, letter_pressed, usable_frames, interval):
        """
        Handles when a letter is pressed during annotation. Letters supported:
        Left arrow key - unicode Q - skips interval-many frames backward, can go into already labelled frames.
        Right arrow key - unicode S - skips interval-many frames forward, can go into already labelled frames.
        Spacebar - Repeats loop at current frame header.
        r - unicode r - Selects random frame from usable frames.
        p - unicode p - Waits on a label, then takes you to frame who has been predicted as a certain label. Prioritizes
        usable frames.
        Escape key - unicode \x1b - finishes annotating, goes to outro screen.

        :param letter_pressed: Unicode of letter pressed during annotation.
        :param usable_frames: Dataframe of usable frames.
        :param interval: Length of loop, in frames.
        """
        pickle_df = self.pickle_cache
        total_frames = len(self.cap)
        if letter_pressed == ord('r'):
            # Pick random frame from usable frames, and changes frame header by reference.
            self.frame_header = usable_frames.sample()['frame'].iloc[0]

        # This is really hacky, but I can think of no readily available workaround.
        #  Simply calling video_loop will still increment frame_header and will result in undefined behavior.
        #  Decreasing frame_header might cause you to end up at an already labelled frame outside of the bounds of
        #   usable frames, which is also undefined behavior.
        if letter_pressed == ord(' '):
            # This exception will be caught in the loop and causes the annotator to restart the loop.
            raise StopIteration("Spacebar.")
        # Esc pressed. Exception caught in loop.
        if letter_pressed == ord('\x1b'):
            raise KeyboardInterrupt('Escape key pressed.')
        # Left arrow key
        if letter_pressed == ord('Q'):
            print(f'Left arrow key pressed. Frame header: {self.frame_header}')
            self.frame_header -= interval
            if self.frame_header < 0:
                self.frame_header = 0
            print(f'    Frame header changed to: {self.frame_header}')
            return
        # Right arrow key.
        if letter_pressed == ord('S'):
            print(f'    Right arrow key pressed. Frame header: {self.frame_header}')
            # Only increments if wont go over frame count.
            if self.frame_header < total_frames - interval:
                self.frame_header += interval
            print(f'Frame header changed to: {self.frame_header}')
            return
        # Choose random frame with certain annotation.
        if letter_pressed == ord('p'):
            print("p pressed.")
            print("Select which annotation you would like to predict:")
            usr_in = cv2.waitKey(0)
            print(f'{usr_in} selected.')
            if usr_in not in list(self.pick_keys.keys()):
                print("Invalid key pressed. Restarting loop.")
                raise KeyError

            behavior = self.BEHAVIOR_LABELS.get(self.pick_keys.get(usr_in))
            print(f'Looking for behavior: {behavior}')
            annotation = self.pick_keys.get(usr_in)
            # All frames that are predicted to be annotation
            predicted_frames = self.csv_df.loc[self.csv_df['pred'] == annotation]['frame']
            if predicted_frames.empty:
                print("No annotations of that type have been predicted.")
                raise KeyError
            # All usable frames that are predicted to be annotation
            usable_predicted_frames = usable_frames[usable_frames['frame'].isin(predicted_frames)]

            if usable_predicted_frames.empty:
                print("No predictions of that type have yet to be annotated, showing previously annotated frames"
                      " of that prediction:")
                self.frame_header = predicted_frames.sample()['frame']
                return
            # Set header to random frame value in set of frames predicted to be annotation
            self.frame_header = usable_predicted_frames.sample()['frame'].values[0]
            return

    def label(self, key_pressed, usable_frames, interval):
        """
        Handles when a number corresponding to a label is pressed
        :param key_pressed: Unicode key pressed
        :param usable_frames: Dataframe of usable frames (not labelled previously)
        :param interval: Length of interval to loop through from frame header
        """
        label_from_key = self.BEHAVIOR_LABELS.get(self.pick_keys.get(int(key_pressed)))
        # Make a dataframe of size 'interval' with one column being range(frame_header, interval),
        #  other column is label.
        frames_labelled = list(range(self.frame_header, self.frame_header + interval))
        data = np.transpose([frames_labelled, [label_from_key] * interval])
        data_df = pd.DataFrame(data, columns=['frame', 'label'])
        data_df['frame'] = data_df['frame'].astype('int64')
        # Update the pickle_cache (yet to be saved) with the labelled frames
        self.pickle_cache = self.pickle_cache.append(data_df, ignore_index=True)
        print(self.pickle_cache)

        indices_in_usable_labelled = usable_frames[usable_frames['frame'].isin(frames_labelled)].index
        # Drop labelled frames from usable frames after incrementing header.
        if indices_in_usable_labelled.size != 0:
            usable_frames.drop(indices_in_usable_labelled, inplace=True)

        self.calculate_header(interval, usable_frames)

    def organize_files(self):

        file_org = self.metrics.determine_folder_hierarchy(self.main_path, rsync_path=self.rsync_path)
        if not file_org:
            sys.exit('Error: Directories not organized properly. See documentation for further help')

        video_files = self.metrics.check_video_files(rsync_path=self.rsync_path)
        if video_files is not None:
            video_file, video_rsync, test_or_train = self.display.pick_file(video_files, self.main_path,
                                                                            self.rsync_path)
            self.dataset_test_train = test_or_train
            subprocess.run("rsync {}:{} {}".format(self.address, self.main_path + "/config.yaml",
                                                   self.rsync_path + "/config_yaml"),
                           shell=True)
            boot_round = self.metrics.calculate_config(self.rsync_path + '/config.yaml')
            if len(video_files) == 1:
                print("No videos in directory. Add videos and re-run artemis_annotation")
        else:
            video_file, test_or_train = self.display.choose_local_file(self.main_path + "/videos_not_done/")
            boot_round = self.metrics.calculate_config(self.main_path + '/config.yaml')

        video_path, pickle_path, pickle_rsync_path, csv_path, csv_rsync_path = self.metrics.create_file_names(
            video_file,
            self.main_path,
            test_or_train,
            boot_round,
            rsync_path=self.rsync_path)

        if self.encoding is None:
            self.encoding = artemis_annotation_calculation.detect_encoding(csv_path)

        return video_path, pickle_path, pickle_rsync_path, csv_path, csv_rsync_path

    def get_usable_dataframe(self, video_path, final_pickle_path, final_csv_path=None):
        """
        Gets the frames that are not yet labelled by looking through the pickle files.
        Sets attribute variables of path to csv prediction file and csv data frame.
        :param video_path: Path to video to be labelled.
        :param final_pickle_path: Path or rsync path to pickle file for video.
        :param final_csv_path: Path or rsync path to csv prediction for video.
        :return: Data frame ['frame', 'label'] of frames that have not been labelled.
        """

        # Set type of frame to int64 so that duplicates can be detected.
        type_replaced = self.pickle_data['frame'].astype(dtype='int64')
        self.pickle_data = self.pickle_data.assign(frame=type_replaced)

        # Update CSV Path to metrics.
        self.metrics.set_csv_path(final_csv_path)
        total_frames = artemis_annotation_calculation.calculate_frames(video_path)

        # Total number of frames
        df_total_frames = pd.Series(range(0, total_frames + 1))

        # Find frames that have yet been annotated.
        not_analyzed = pd.concat([self.pickle_data['frame'], df_total_frames]).drop_duplicates(keep=False).reset_index()
        not_analyzed = not_analyzed.rename(columns={'index': 'frame'})
        return not_analyzed

    def load_data(self, video_path, pickle_path, csv_path):
        """
        Loads video, pickle file, csv prediction file into attribute variables.
        :param csv_path: Path to csv file containing predictions.
        :param pickle_path: Path to pickle file containing labels.
        :param video_path: Path to video.
        """
        self.cap = pims.PyAVReaderTimed(video_path)

        total_frames = artemis_annotation_calculation.calculate_frames(video_path)
        # Save dataframe of predictions as attribute, otherwise fill it with 'N/A' entries.
        self.csv_path = csv_path
        # This try-catch statement tries to read from an existing csv file, otherwise makes a default dataframe.
        try:
            self.csv_df = pd.read_csv(self.csv_path, encoding=self.encoding, dtype='int64')
        except:
            self.csv_df = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
        # If not all frames are labelled, fill the rest with 'N/A'.
        non_labelled_frames = total_frames - len(self.csv_df)
        non_labelled_frames_df = pd.Series(range(1, non_labelled_frames + 1))
        # Note: error will occur if names of columns on dict below don't match column names of csv df.
        default_data = {'frame': non_labelled_frames_df, 'pred': [10] * non_labelled_frames}
        if non_labelled_frames > 0:
            default_data_df = pd.DataFrame(default_data)
            self.csv_df.columns = ['frame', 'pred']
            self.csv_df = self.csv_df.append(default_data_df)

        self.display.setup_video_properties(video=self.cap, csv_df=self.csv_df)
        # Read pickle file
        if os.path.getsize(pickle_path) > 0:
            self.pickle_data = pd.read_pickle(pickle_path)
        else:
            print(f"Pickle path {pickle_path} empty.")
        self.pickle_data.columns = ['frame', 'label']
        return

    def annotate_video(self, usable_frames, pickle_path, predictions_csv, interval=None, fps=30):
        """
        Begins a loop at the first usable frame.
        Starts by displaying intro, then calls video loop at header of usable frames.
        :param fps: Frames per second to loop videos at
        :param interval: Amount of frames to loop through from header.
        :param usable_frames: Dataframe of frames that are yet labelled.
        :param pickle_path: Path to pickle file
        :param predictions_csv: Path to csv prediction file.
        """

        if interval is None:
            interval = self.interval

        predictions = self.csv_df
        self.frames_labelled_in_session = len(usable_frames)
        self.display.intro()
        # display interval with prediction
        video = self.cap
        # First usable frame
        self.frame_header = usable_frames['frame'].iloc[0]
        annotate = True
        while annotate:
            # Loop at header.
            self.display.video_loop(video=video, start=self.frame_header, csv_path=predictions_csv, interval=interval,
                                    fps=fps)
            usr_in = cv2.waitKeyEx(0)
            masked = artemis_annotation_calculation.mask_keycode(usr_in)
            try:
                # PyCharm might be nasty here, but this works as long as the arguments to the functions
                #  are EXACTLY the same - 3 arguments each (as of now).
                #  Also note that this is where the frame header gets incremented.
                self.handle_input.get(masked)(masked, usable_frames, interval)
            except TypeError:
                print(f"Incorrect key pressed. Key: {usr_in} - Masked: {masked}")
                print(traceback.format_exc())
                continue
            except StopIteration:
                print("Space bar clicked.")
                continue
            except KeyError as e:
                tb = traceback.format_exc()
                print(tb)
            except KeyboardInterrupt:
                print("ESC pressed. Done with video, proceeding to save.")
                cv2.destroyAllWindows()
                print(traceback.format_exc())
                annotate = False
            # Increment header by interval, except we only look over frames that are usable.
            except IndexError as e:
                print("IndexError: Reached end of usable frames.")
                print(traceback.format_exc())
                annotate = False
        frames_not_annotated = len(usable_frames)
        total_frames = len(video)
        self.frames_labelled_in_session = self.frames_labelled_in_session - len(usable_frames)
        percent = 100 * (self.frames_labelled_in_session + len(self.pickle_data)) / total_frames
        user_input = self.display.done_with_video(total_frames, self.pickle_data, percent=percent)
        self.save_pickle_and_exit(pickle_path, user_input=user_input)

    def calculate_header(self, interval, usable_frames):
        """
        Increments frame header to next usable frame. If no usable frame at header + interval,
        goes to closest frame to the destination header.
        :param interval: Amount of frames to loop through from header.
        :param usable_frames: Dataframe of usable frames.
        """
        # The below shenanigans are to prevent labelling already labelled frames.
        tmp_header = self.frame_header
        new_header = self.frame_header + interval

        frame_at_new_header = usable_frames[usable_frames['frame'] == (new_header)]

        # If the new header does not have a frame in usable frames, we take use the closest frame.
        if frame_at_new_header.empty:
            smaller_than = usable_frames[usable_frames['frame'] < new_header]
            bigger_than = usable_frames[usable_frames['frame'] >= new_header]

            if smaller_than.empty and bigger_than.empty:
                raise IndexError("No usable frames left.")
            elif smaller_than.empty:
                print('No frames beneath.')
                closest_bigger = bigger_than.iloc[0]['frame']
                self.frame_header = closest_bigger
            elif bigger_than.empty:
                print('No frames above.')
                closest_lower = smaller_than.iloc[-1]['frame']
                self.frame_header = closest_lower
            else:
                closest_lower = smaller_than.iloc[-1]['frame']
                closest_bigger = bigger_than.iloc[0]['frame']
                closest_overall = min(abs(closest_bigger - new_header), abs(closest_lower - new_header))
                self.frame_header = closest_overall
        else:
            self.frame_header = frame_at_new_header.iloc[0]['frame']
        #  the first and last of each. Check which is closer, and set frame header to that.
        print(f'HEADER: {tmp_header} --> {self.frame_header}')

    def save_pickle_and_exit(self, pickle_path, user_input=None):
        """
        Saves pickle if input given is characters 's' or 'm', otherwise aborts save.
        If user_input is none, saves. This way it can be called without user input somewhere in the code to save.
        :param pickle_path: Path to pickle file for video.
        :param user_input: Input of user - a character
        """
        if user_input != ord('s') and user_input != ord('m') and user_input is not None:
            print("Not 's' or 'm', not saving progress.")
            return

        # Copy frame column and type-cast to int. It is duplicated to prevent propagation to other pandas objects
        # Copy frame column and type-cast to int. It is duplicated to prevent propagation to other pandas objects
        type_replaced = self.pickle_data['frame'].astype(dtype='int64')
        self.pickle_data = self.pickle_data.assign(frame=type_replaced)
        self.pickle_cache.sort_values(by='frame', inplace=True)
        self.pickle_data = self.pickle_data.append(self.pickle_cache)
        self.pickle_data.drop_duplicates(subset=['frame'], inplace=True, keep='last')
        # Sort the entire pickle file
        self.pickle_data.sort_values(by='frame', inplace=True)
        pd.to_pickle(self.pickle_data, pickle_path)
        # Reset pickle. Not very useful, but why not I guess.
        self.pickle_cache = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
        print(f'You labelled {self.frames_labelled_in_session} frames this session.')
        print(f'There are {len(self.pickle_data)} frames labelled total.')
        sys.exit()
