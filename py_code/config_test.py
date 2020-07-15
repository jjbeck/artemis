import yaml
import pandas as pd
from operator import add
import re
import numpy as np

def unpack_metrics(drink=[0, 0], eat=[0, 0], groom=[0, 0],
                   hang=[0, 0], sniff=[0, 0], rear=[0, 0], rest=[0, 0], walk=[0, 0], eathand=[0, 0], none=[0, 0]):
    return {"drink": drink, "eat": eat, "groom": groom, "hang": hang,
            "sniff": sniff, "rear": rear, "rest": rest, "walk": walk, "eathand": eathand}


def update_metrics(new_test_set_beh, old_test_set_beh, new_train_set_beh = None, old_train_set_beh = None, sum_bouts):
    # create empyty dictionaries to append bouts and samples to
    train = True
    update_bouts= {'old': {"drink": [0, 0],
               "eat": [0, 0],
               "groom": [0, 0],
               "hang": [0, 0],
               "sniff": [0, 0],
               "rear": [0, 0],
               "rest": [0, 0],
               "walk": [0, 0],
               "eathand": [0, 0]},
                   'new': {"drink": [0, 0],
                           "eat": [0, 0],
                           "groom": [0, 0],
                           "hang": [0, 0],
                           "sniff": [0, 0],
                           "rear": [0, 0],
                           "rest": [0, 0],
                           "walk": [0, 0],
                           "eathand": [0, 0]}
                   }
    update_bouts_boot = {'old': {"drink": [0, 0],
               "eat": [0, 0],
               "groom": [0, 0],
               "hang": [0, 0],
               "sniff": [0, 0],
               "rear": [0, 0],
               "rest": [0, 0],
               "walk": [0, 0],
               "eathand": [0, 0]},
                   'new': {"drink": [0, 0],
                           "eat": [0, 0],
                           "groom": [0, 0],
                           "hang": [0, 0],
                           "sniff": [0, 0],
                           "rear": [0, 0],
                           "rest": [0, 0],
                           "walk": [0, 0],
                           "eathand": [0, 0]}
                   }
    try:
        sum_bouts = sum_bouts['old']
        video_time = 'old'
    except:
        sum_bouts = sum_bouts['new']
        video_time = 'new'

    bout = pd.DataFrame()
    sample = pd.DataFrame()

    for key in new_test_set_beh.keys():
        a = unpack_metrics(**new_test_set_beh[key])
        for beh in a.keys():
            update_bouts['new'][beh] = list(map(add, update_bouts['new'][beh], a[beh]))

    for key in old_test_set_beh.keys():
        a = unpack_metrics(**old_test_set_beh[key])
        for beh in a.keys():
            update_bouts['old'][beh] = list(map(add, update_bouts['old'][beh], a[beh]))

    for key in sum_bouts.keys():
        a = unpack_metrics(**sum_bouts[key])
        for beh in a.keys():
            update_bouts[video_time][beh] = list(map(add,update_bouts[video_time][beh], a[beh]))

    return update_bouts

def calculate_bouts_samples(csv,file):
    final_test = {}
    test_vers_update = {}
    sample_total = {"drink": 0,
                    "groom": 0,
                    "eat": 0,
                    "hang": 0,
                    "sniff": 0,
                    "rear": 0,
                    "rest": 0,
                    "walk": 0,
                    "eathand": 0,
                    "none": 0}
    pred_sum = {}

    csv.sort_values(by='frame', inplace=True)
    csv = csv[csv.pred != 'none']
    test_beh = (csv)
    file = file[file.rfind('/') + 1:]
    test_experiment = (file[:file.find('_')])
    matches = re.finditer("_", file)
    matches_positions = [match.start() for match in matches]
    test_exp_version = (file[matches_positions[1] + 1:matches_positions[2]])

    frame_one = csv['frame'].iloc[0]
    frame_lst = csv['frame'].iloc[0]
    for index, row in csv.iterrows():
        if row['frame'] - frame_lst > 1:
            if frame_lst - frame_one >= 64:
                pred_comp = csv.loc[csv['frame'] == frame_one]
                pred_comp = pred_comp['pred'].to_string(index=False).strip()
                for frame in np.arange(frame_one + 1, frame_lst - 63):
                    pred_lst = csv.loc[csv['frame'] == frame]
                    pred_lst = pred_lst['pred'].to_string(index=False).strip()
                    sample_total[pred_lst] += 1
                    if pred_comp != pred_lst:
                        if pred_comp in pred_sum:
                            pred_sum[pred_comp] += 1
                            pred_comp = csv.loc[csv['frame'] == frame]
                            pred_comp = pred_comp['pred'].to_string(index=False).strip()
                        else:
                            pred_sum[pred_comp] = 1
                            pred_comp = csv.loc[csv['frame'] == frame]
                            pred_comp = pred_comp['pred'].to_string(index=False).strip()

                if pred_comp in pred_sum:
                    pred_sum[pred_comp] += 1

                else:
                    pred_sum[pred_comp] = 1

                frame_one = row['frame']
                frame_lst = row['frame']
            else:
                frame_one = row['frame']
                frame_lst = row['frame']
        else:
            frame_lst = row['frame']

    if frame_lst - frame_one >= 64:
        pred_comp = csv.loc[csv['frame'] == frame_one]
        pred_comp = pred_comp['pred'].to_string(index=False).strip()
        for frame in np.arange(frame_one + 1, frame_lst - 63):
            pred_lst = csv.loc[csv['frame'] == frame]
            pred_lst = pred_lst['pred'].to_string(index=False).strip()
            if pred_comp != pred_lst:
                if pred_comp in pred_sum:
                    pred_sum[pred_comp] += 1
                    pred_comp = csv.loc[csv['frame'] == frame]
                    pred_comp = pred_comp['pred'].to_string(index=False).strip()
                else:
                    pred_sum[pred_comp] = 1
                    pred_comp = csv.loc[csv['frame'] == frame]
                    pred_comp = pred_comp['pred'].to_string(index=False).strip()
        if pred_comp in pred_sum:
            pred_sum[pred_comp] += 1

        else:
            pred_sum[pred_comp] = 1

    test_update_params = {}
    test_total = {"drink": [0, 0],
                  "groom": [0, 0],
                  "eat": [0, 0],
                  "hang": [0, 0],
                  "sniff": [0, 0],
                  "rear": [0, 0],
                  "rest": [0, 0],
                  "walk": [0, 0],
                  "eathand": [0, 0],
                  "none": [0, 0]}

    for key_test in pred_sum.keys():
        test_total[key_test][0] += pred_sum[key_test]

    for key_test in sample_total.keys():
        test_total[key_test][1] += sample_total[key_test]

    if test_exp_version in test_vers_update:
        if test_experiment in test_vers_update[test_exp_version]:
            test_update_params[test_experiment] = test_total
            for key_tests in test_update_params[test_experiment].keys():
                test_vers_update[test_exp_version][test_experiment][key_tests][0] += test_total[key_tests][0]
                test_vers_update[test_exp_version][test_experiment][key_tests][1] += test_total[key_tests][1]

        else:
            test_vers_update[test_exp_version][test_experiment] = {0: 0}
            test_vers_update[test_exp_version][test_experiment] = test_total
    else:
        test_update_params[test_experiment] = test_total
        test_vers_update[test_exp_version] = test_update_params
    return test_vers_update

with open( "/home/jordan/Desktop/andrew_nih/Annot/config_yaml") as file:
    config_param = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
boot_round = config_param["Boot Round"]
new_test_set_beh = config_param["Number of behaviors for Test Set"]["new"]
old_test_set_beh = config_param["Number of behaviors for Test Set"]["old"]
try:
    new_train_set_beh= config_param["Boot 1 Behaviors"]["new"]
    old_train_set_beh= config_param["Boot 1 Behaviors"]["old"]
    new_train_set_beh = {"new": new_train_set_beh}
    old_train_set_beh = {"old": old_train_set_beh}
except:
    print("no new")
pk_file = pd.read_pickle('/home/jordan/Desktop/Annot/pickle_files/test/Alc_B-W2-04-Notdrinking_old_video_2019Y_04M_12D_08h_21m_24s_cam_17202339-0000_test.p')

new_test_set_beh = {"new":new_test_set_beh}
old_test_set_beh = {"old":old_test_set_beh}
pk_bouts = calculate_bouts_samples(pk_file,'/home/jordan/Desktop/Annot/pickle_files/test/Alc_B-W2-04-Notdrinking_old_video_2019Y_04M_12D_08h_21m_24s_cam_17202339-0000_test.p')
update_bouts = update_metrics(new_test_set_beh,old_test_set_beh,new_train_set_beh, old_train_set_beh, pk_file)
print(new_test_set_beh)
print(old_test_set_beh)
print(pk_bouts)
print(update_bouts)
final_test={}
config_param = {"Boot Round": 1, "Main Path": '/home/jordan/Desktop/andrew_nih/Annot'}
final_test["Number of behaviors for Test Set"] = update_bouts
with open("/home/jordan/Desktop/andrew_nih/Annot/config_yaml", 'w') as file:
    documents = yaml.dump(config_param, file)
    documents = yaml.dump(final_test, file)

file.close()


