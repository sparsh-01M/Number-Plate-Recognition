# Number-Plate-Recognition
This project is used to detect number plate from vehicles from a live video feed.

# Approach: 

Find all the contours in the image.
Find the bounding rectangle of every contour.
Compare and validate the sides ratio and area of every bounding rectangle with an average license plate.
Apply image segmentation in the image inside the validated contour to find characters in it.
Recognize characters using an OCR.

Methodology: 
1. To reduce the noise we need to blur the input Image with Gaussian Blur and then convert it to grayscale.  
2. Find vertical edges in the image.
3. To reveal the plate we have to binarize the image. For this apply Otsu’s Thresholding on the vertical edge image. In other thresholding methods, we have to choose a threshold value to binarize the image but Otsu’s Thresholding determines the value automatically.
4. Apply Closing Morphological Transformation on the thresholded image. Closing is useful to fill small black regions between white regions in a thresholded image. It reveals the rectangular white box of license plates.
5. To detect the plate we need to find contours in the image. It is important to binarize and morph the image before finding contours so that it can find a more relevant and less number of contours in the image. If you draw all the extracted contours on the original image, it would look like this: 
