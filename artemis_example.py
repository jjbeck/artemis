from artemis_v3_pickle_change.py import show_prediction

a = show_prediction('/home/jordan/Desktop/nihgpppipe/Annot','/home/jordan/Desktop/nihgpppipe/Annot',
                    '/home/jordan/Desktop/nihgpppipe/Annot')
a.show_intro()
a.load_video_organize_dir()
last_frame = a.determine_last_frame()
a.loop_video(last_frame)
