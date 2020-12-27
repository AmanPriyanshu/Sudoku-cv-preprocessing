import cv2
import numpy as np
import os

def bgr2gray(img):
	b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	img_grayscale = 0.2126*r + 0.7152*g + 0.0722*b
	img_grayscale = img_grayscale.astype(np.uint8)
	return img_grayscale

def apply_filter(img, filter_arr, pad=True):
	assert filter_arr.shape[0] == filter_arr.shape[1], "Not a Square Filter."
	reduction = filter_arr.shape[0]//2
	if pad:
		img_padded = np.zeros((img.shape[0]+2*reduction, img.shape[1]+2*reduction))
		img_padded[reduction:-reduction, reduction:-reduction] = img
		img = img_padded
	filtered = np.zeros((img.shape[0]-2*reduction, img.shape[1]-2*reduction))
	for row_new, row in enumerate(range(reduction, img.shape[0]-reduction)):
		for col_new, col in enumerate(range(reduction, img.shape[1]-reduction)):
			filtered[row_new][col_new] = np.sum(img[row-reduction:row+reduction+1, col-reduction:col+reduction+1] * filter_arr)
	filtered = filtered.astype(np.uint8)
	return filtered

def gaussian_blur(img, size=3):
	sigma = ((size - 1)/2 - 0.5)/4 
	sigma = sigma + 1e-5
	filter_size = size
	gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
	m = filter_size//2
	n = filter_size//2
	for x in range(-m, m+1):
		for y in range(-n, n+1):
			gaussian_filter[x+m, y+n] = (1/ (2*np.pi*(sigma**2)) )*(np.exp(-(x**2 + y**2)/(2* sigma**2)))
	return apply_filter(img, gaussian_filter)

def threshold(img, size=3, c=2):
	average_filter = np.full((size, size), (1/(size**2)))
	thresholded = apply_filter(img, average_filter)
	thresholded = (img < thresholded - c).astype(np.uint8) * 255
	return thresholded

def preprocess(img):
	os.system('mkdir preprocessed')
	cv2.imwrite('./preprocessed/original.png', img)
	img_grayscale = bgr2gray(img)
	cv2.imwrite('./preprocessed/grayscale.png', img_grayscale)
	filtered = gaussian_blur(img_grayscale, 9)
	cv2.imwrite('./preprocessed/gaussian_filtered.png', filtered)
	thresholded = threshold(filtered, size=15, c=5)
	cv2.imwrite('./preprocessed/thresholded.png', thresholded)
	return thresholded

if __name__ == '__main__':
	img = cv2.imread('./example.jpeg')
	img = preprocess(img)
	