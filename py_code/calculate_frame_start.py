import argparse
import pandas as pd
import glob
import numpy as np
from __future__ import print_function

def determine_last_frame(pickle_path,video_file,test_or_train, boot_round):
    """
    Determines last frame. Basically a save mechanism so you don't have to start over.
    :return:
    start_frame to pass annotation gui.
    """
    frames_analyzed = []
    exp_frames_analyzed_list = []
    frame_arr = np.arange(80, (108000 + 1))
    total_frames = pd.DataFrame(data=frame_arr, columns=['frame'])
    start_frame = []
    prediction_state = False

    for file in glob.glob(pickle_path + "/train" + video_file[video_file.rfind('/'):-4] + '*'):

        annot_pickle = pd.read_pickle(file)
        annot_pickle.sort_values(by='frame', inplace=True)
        exp_frames_analyzed_list.append(annot_pickle)
        frames_analyzed.append(len(annot_pickle.index))
    for file in glob.glob(pickle_path + "/test" + video_file[video_file.rfind('/'):-4] + '*'):
        annot_pickle = pd.read_pickle(file)
        annot_pickle.sort_values(by='frame', inplace=True)
        exp_frames_analyzed_list.append(annot_pickle)
        frames_analyzed.append(len(annot_pickle.index))
    try:
        exp_frames_analyzed = pd.concat(exp_frames_analyzed_list, ignore_index=True)
        exp_frames_analyzed['frame'] = exp_frames_analyzed['frame'].astype('int32')
        non_analyzed_frames = pd.concat([total_frames, exp_frames_analyzed, exp_frames_analyzed],
                                             sort=True).drop_duplicates(subset=['frame'], keep=False)

    except:
        non_analyzed_frames = total_frames
        pass

    if test_or_train == 'test':
        try:
            annot_pickle = pd.read_pickle(
                pickle_path + "/test" + video_file[video_file.rfind('/'):-4] + '_test.p')
            annot_pickle.sort_values(by='frame', inplace=True)
            annot_pickle.drop_duplicates(subset=['frame'])
            pickl_pres = True
        except:
            annot_pickle = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
            pickl_pres = False
            pass
    else:
        try:
            annot_pickle = pd.read_pickle(
                pickle_path + "/train" + video_file[video_file.rfind('/'):-4] + '_boot{}.p'.format(
                    boot_round))
            annot_pickle.sort_values(by='frame', inplace=True)
            annot_pickle.drop_duplicates(subset=['frame'])
            pickl_pres = True
        except:
            annot_pickle = pd.DataFrame(columns=['frame', 'pred'], dtype='int64')
            pickl_pres = False
            pass
    self_start = (non_analyzed_frames['frame'].iloc[0])
    print(f"Your current pickle file has {len(annot_pickle)} frames annotated")

    return self_start

def main():

    parser = argparse.ArgumentParser(description="Add main path and frame length for video loop")
    parser.add_argument("-mp", "-main_path", help="Directory where you want all files associated with artemis annotations saved. This will create a folder called Annot whcih will hold all files,"
                                                  "Different experiments can be housed in separate folders under different Annot folder")
    parser.add_argument("-vf", "-video_file",
                        help="Directory where you want all files associated with artemis annotations saved. This will create a folder called Annot whcih will hold all files,"
                             "Different experiments can be housed in separate folders under different Annot folder")
    parser.add_argument("-tt", "-test_train",
                        help="Directory where you want all files associated with artemis annotations saved. This will create a folder called Annot whcih will hold all files,"
                             "Different experiments can be housed in separate folders under different Annot folder")

    parser.add_argument("-br", "-boot_round",
                        help="Directory where you want all files associated with artemis annotations saved. This will create a folder called Annot whcih will hold all files,"
                             "Different experiments can be housed in separate folders under different Annot folder")

    args = parser.parse_args()
    return args.mp, args.vf, args.tt, args.br

if __name__ == "__main__":

    mp,vf, tt, br =  main()
    mp = mp + "Annot/pickle_files"
    print(determine_last_frame(mp,vf, tt, br))