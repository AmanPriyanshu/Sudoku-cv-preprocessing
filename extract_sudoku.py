from preprocessing import preprocess, threshold
import numpy as np
import cv2
import os


def find_quadrilaterals(img):
	contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  

	#Sorted Contours, that way we get the largest one back when we approximate using poly in the next stage

	for c in contours:
		perimeter = cv2.arcLength(c, True)
		polynomial = cv2.approxPolyDP(c, 0.02 * perimeter, True)
		if len(polynomial) == 4:
			return polynomial

def order_corner_points(corners):
	corners = [(corner[0][0], corner[0][1]) for corner in corners]
	return corners[1], corners[0], corners[3], corners[2]

def transform(img, corners):
	ordered_corners = order_corner_points(corners)
	top_left, top_right, bottom_right, bottom_left = ordered_corners

	# Finding all 4 dimensions of quadrilateral, since shape we made sure is 4

	width_bottom = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
	width_top = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
	width = int(max(width_bottom, width_top))
	height_right = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
	height_left = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
	height = int(max(height_left, height_right))

	dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],[0, height - 1]], dtype="float32")

	ordered_corners = np.array(ordered_corners, dtype=np.float32)

	grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
	return cv2.warpPerspective(img, grid, (width, height))

def cropped_and_transformed(img):
	img_processed = preprocess(img)
	corners = find_quadrilaterals(img_processed)
	transformed = transform(img, corners)
	cv2.imwrite('./preprocessed/cropped_and_transformed.png', transformed)
	return transformed

def create_image_grid(img):

	img = cropped_and_transformed(img)

	os.system('mkdir sudoku')

	### Assuming some amount of equality among rows and column dimensions ###

	grid = np.copy(img)
	edge_h = np.shape(grid)[0]
	edge_w = np.shape(grid)[1]
	celledge_h = edge_h // 9
	celledge_w = edge_w // 9
	grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
	grid = cv2.bitwise_not(grid, grid)

	tempgrid = []
	for i in range(celledge_h, edge_h + 1, celledge_h):
		for j in range(celledge_w, edge_w + 1, celledge_w):
			rows = grid[i - celledge_h:i]
			tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

	finalgrid = []
	for i in range(0, len(tempgrid) - 8, 9):
		finalgrid.append(tempgrid[i:i + 9])

	for i in range(9):
		for j in range(9):
			finalgrid[i][j] = 255 - np.array(finalgrid[i][j])
			finalgrid[i][j] = threshold(finalgrid[i][j], c=5, size=5)
			cv2.imwrite(str("./sudoku/cell" + str(i) + str(j) + ".jpg"), finalgrid[i][j])

	return finalgrid


if __name__ == '__main__':
	img = cv2.imread('./example.jpg')
	sudoku = create_image_grid(img)