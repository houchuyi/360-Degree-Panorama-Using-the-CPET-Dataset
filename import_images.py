import shutil

src = 'C:/Users/skyho/Desktop/YEAR 4/ROB501/Assignments/Project/data/omni_image'
dst = './data/omni_image'
frame = '/frame000592_2018_09_04_17_45_39_905367.png'

for i in range(0,10):
    shutil.copy2(src+str(i)+frame, dst+str(i)+'/image.png') # complete target filename given

print("Copy images completed")
