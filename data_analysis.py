import load_data
import matplotlib.pyplot as plt
import numpy as np
import cv2

# steer = load_data.load_steer_angles(
#     ['../data/driving_log.csv'])#, '../data0/driving_log.csv', '../data1/driving_log.csv', '../data2/driving_log.csv',
#      # '../data3/driving_log.csv', '../data4/driving_log.csv'])
#
# n_bins = 80
# fig, axes = plt.subplots(nrows=5, ncols=1)
# ax0, ax1, ax2, ax3, ax4 = axes.flatten()
#
# ax0.hist(steer['center'], n_bins)
# ax0.set_title('center angle histogram')
# ax0.set_xlabel('angle')
# ax0.set_ylabel('count')
# ax0.set_xlim(-1.25, +1.25)
# ax0.grid(True)
#
# ax1.hist(steer['left'], n_bins)
# ax1.set_title('left angle histogram')
# ax1.set_xlabel('angle')
# ax1.set_ylabel('count')
# ax1.set_xlim(-1.25, +1.25)
# ax1.grid(True)
#
# ax2.hist(steer['right'], n_bins)
# ax2.set_title('right angle histogram')
# ax2.set_xlabel('angle')
# ax2.set_ylabel('count')
# ax2.set_xlim(-1.25, +1.25)
# ax2.grid(True)
#
# ax3.hist(steer['flipped'], n_bins)
# ax3.set_title('flipped angle histogram')
# ax3.set_xlabel('angle')
# ax3.set_ylabel('count')
# ax3.set_xlim(-1.25, +1.25)
# ax3.grid(True)
#
# total = np.concatenate((steer['center'], steer['left'], steer['right'], steer['flipped']), axis=0)
# ax4.hist(total, n_bins)
# ax4.set_title('all angle histogram')
# ax4.set_xlabel('angle')
# ax4.set_ylabel('count')
# ax4.set_xlim(-1.25, +1.25)
# ax4.grid(True)
# fig.tight_layout()
# plt.show()
# print(np.mean(total))
# print(np.std(total))


#original image
original_img = cv2.imread('center_2017_02_25_23_58_39_725.jpg')
print('original image shape: {}'.format(original_img.shape))
cv2.imshow("original", original_img)
cv2.imwrite('original_image.jpg', original_img)

#crop image
cropped_img = original_img[50:140, :]
print('cropped image shape: {}'.format(cropped_img.shape))
cv2.imshow('cropped image', cropped_img)
cv2.imwrite('cropped_image.jpg', cropped_img)

#flipped image
flipped_img = cv2.flip(cropped_img, 1)
cv2.imshow('flipped image', flipped_img)
cv2.imwrite('flipped_image.jpg', flipped_img)
#
# #resize image
# resized_img = cv2.resize(cropped_img, (int(cropped_img.shape[1]/2.0), int(cropped_img.shape[0]/2)))
# print('resized image shape: {}'.format(resized_img.shape))
# cv2.imshow('resized image', resized_img)


cv2.waitKey(0)
cv2.destroyAllWindows()



