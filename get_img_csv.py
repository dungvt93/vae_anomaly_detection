import csv
import os
import shutil

f = open('../20190829_dataset/bonus_1e-10/note.csv', 'r')
sorce_folder = '../20190829_dataset/result/origin/'
des_folder= '../20190829_dataset/bonus_1e-10/'

reader = csv.reader(f)
for row in reader:
    print(os.path.basename(row[0]))
    shutil.copyfile(sorce_folder+os.path.basename(row[0]),des_folder+os.path.basename(row[0]))

f.close()
