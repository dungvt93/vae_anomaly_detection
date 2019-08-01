import cv2
import os


source_folder = './test_origin/'
target_folder = './train/'

for i in range(9):
	if i == 3:
		if not os.path.exists(target_folder + '{:02d}'.format(i)):
			print('Make folder' + target_folder + '{:02d}'.format(i))
			os.mkdir(target_folder + '{:02d}'.format(i))

num_pins = []
for file_name in os.listdir(source_folder):
	img = cv2.imread(source_folder + file_name)
	num_pins.append(round(img.shape[1] / 90))

files = os.listdir(source_folder)
for m in range(9):
	count = 0
	if m==3:
		for i in range(len(files)):
			file = files[i]
			img = cv2.imread(source_folder + file)
			# size = (num_pins[i]*64, 9*64)
			size = (64, 9*64)
			img = cv2.resize(img, size)
			img_height, img_width, _ = img.shape
			# for n in range(num_pins[i]):
			for n in range(1):
				start_y = max(0,64*m)
				end_y = min(img_height, 64*(m+1))
				start_x =  max(0, 64*n)
				end_x = min(img_width, 64*(n+1))
				sub_img = img[start_y: end_y, start_x: end_x]
				cv2.imwrite(target_folder + '{:02d}'.format(m) + '/' + '{:06d}.png'.format(count), sub_img)
				print(target_folder + '{:02d}'.format(m) + '/' + '{:06d}.png'.format(count))
				count = count + 1
