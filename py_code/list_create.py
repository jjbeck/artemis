import glob
import pandas as pd
import numpy as np
import pickle
pick = pd.read_pickle('/home/jordan/Desktop/andrew_nih/Annot/pickle_files/test_valid_2_sami/Trap2_FC-A-1-12-Postfearret_new_video_2019Y_02M_18D_18h_40m_43s_cam_17202339-0000_test_sami.p')

pick_list = []
for idx in np.arange(0,108000):
    a = pick.loc[pick['frame'] == idx]
    a = a['pred'].to_string(index=False).strip()
    if a == 'none':
        pick_list.append(a)
    else:
        pick_list.append(a)

with open('/home/jordan/Desktop/andrew_nih/Annot/pickle_files/Trap2_FC-A-1-12-Postfearret_new_video_2019Y_02M_18D_18h_40m_43s_cam_17202339-0000_test_sami.p', 'wb') as f:
    pickle.dump(pick_list, f)
print(pick_list)

