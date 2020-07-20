
import glob
test = False
file_path = "/home/jordan/Desktop/andrew_nih/Annot/"
pickle_test = file_path + "pickle_files/test/"
pickle_train = file_path + "pickle_files/train/"
print(pickle_train)
embs_path = file_path + "embs/"

for file in glob.glob(pickle_test + '*'):
    print(file)
    print(embs_path + file[file.rfind("/")+1:-7] + ".p")



