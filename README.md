# Sudoku-cv-preprocessing
Preprocessing Sudoku to be ready for machine learning detection using simple image filters and detection algorithms.
The idea here, is to create a preprocessing pipeline which produces cell-wise images from a sudoku image. It has simple implementations of gaussian blur, thresholding, binarizing among others from scratch; and also the use of Contours and Polynoimal detection using cv2.

## To Run:
To run the code simply execute: `python extract_sudoku.py`. Two folders will be generated both which are self-explanatory. There contents will be as follows:

```console
./preprocessed/:


27-12-2020  18:53           476,350 cropped_and_transformed.png
27-12-2020  18:53           283,973 gaussian_filtered.png
27-12-2020  18:53           332,409 grayscale.png
27-12-2020  18:53           955,776 original.png
27-12-2020  18:53            33,447 thresholded.png
```

and `sudoku` contains all images of each cell within the sudoku, aptly named after its location in a 2d array, `cell12`: with `cell` prefix and `1`st row and `2`nd column, if the sudoku were to be plotted as a 2d array.