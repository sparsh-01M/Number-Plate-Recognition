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

# Code Explanation:
Important Parts of the Code
1. Loading and Preprocessing the Image
This section of the code is responsible for loading the image and converting it to grayscale, which is a common preprocessing step in image processing tasks. Grayscale images simplify the computational complexity by reducing the image to a single channel.

# Load the image
image = cv2.imread("image_processing/images/test5.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
plt.show()
Image Loading: cv2.imread reads the input image from the specified path.
Grayscale Conversion: cv2.cvtColor converts the image from BGR to grayscale, simplifying the data for further processing.
Displaying the Image: plt.imshow displays the grayscale image for visual inspection.
2. Edge Detection and Contour Finding
Edge detection helps in identifying the boundaries of objects within the image. This step is crucial for locating the number plate by identifying the contours in the image.

python
Copy code
# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, ksize=(3, 5), sigmaX=0.5) 

# Apply Canny edge detection
edges = cv2.Canny(blurred, 30, 200)

# Display edges
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
plt.show()

# Find contours
key = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(key)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
Gaussian Blur: cv2.GaussianBlur reduces noise and detail in the image, making edge detection more reliable.
Canny Edge Detection: cv2.Canny detects edges in the image, highlighting the boundaries.
Finding Contours: cv2.findContours and imutils.grab_contours are used to identify the contours from the edges detected. The contours are then sorted based on their area to prioritize the largest ones.
3. Extracting and Reading the Number Plate
This part of the code identifies the number plate by finding a rectangular contour, extracts the region of interest, and uses EasyOCR to read the text from the number plate.

python
Copy code
# Initialize location variable
location = None

# Loop through contours to find a rectangle (number plate)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 8:
        location = approx
        break

# If no rectangle is found, print an error message
if location is None:
    print("Number plate contour not found")
else:
    # Draw the contours on a mask
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the masked image
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

    # Crop the image to the region of the number plate
    x, y = np.where(mask == 255)
    x1, y1 = (np.min(x), np.min(y))
    x2, y2 = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # Display the cropped image
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB))
    plt.show()

    # Use EasyOCR to read text from the cropped image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    # Print the result
    print(result[1][1])
    
Finding the Number Plate: The code loops through the contours and approximates the contour to find a polygon with 8 vertices, which is assumed to be the number plate.
Masking and Cropping: The number plate contour is drawn on a mask, and the bitwise AND operation extracts the region of interest. The image is then cropped to this region.
Text Recognition: EasyOCR is used to read the text from the cropped image of the number plate, and the result is printed.
These sections are crucial for the functionality of the code, as they cover loading and preprocessing the image, detecting edges and contours, and finally extracting and reading the number plate text.
