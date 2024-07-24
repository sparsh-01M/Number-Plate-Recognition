import cv2
import numpy as np


class PlateFinder: 
	def __init__(self, minPlateArea, maxPlateArea): 
		
		# minimum area of the plate 
		self.min_area = minPlateArea 
		
		# maximum area of the plate 
		self.max_area = maxPlateArea 

		self.element_structure = cv2.getStructuringElement( 
							shape = cv2.MORPH_RECT, ksize =(22, 3)) 
	# function for processing image
	def preprocessing(self, image):
		
		# blurring the image using gaussian blur
		blur = cv2.GaussianBlur(image, (7, 7), 0)
		
		# converting the blur image into grayscale
		grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# finding the vertical edges
		sobelx = cv2.Sobel(grayscale, cv2.CV_8U, 1, 0, ksize = 3)
		
		# Binarizing the image using Otsu's method
		ret2, threshold_img = cv2.threshold(sobelx, 0, 255, 
							cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
		element = self.element_structure 

		# Applying closing morphological transformation to fill the small gaps
		morph_n_thresholded_img = threshold_img.copy() 
		cv2.morphologyEx(src = threshold_img, 
							op = cv2.MORPH_CLOSE, 
							kernel = element, 
							dst = morph_n_thresholded_img) 
			
		return morph_n_thresholded_img 
	
	# function for extracting contours
	def extract_contours(self, after_preprocess): 
		
		contours, _ = cv2.findContours(after_preprocess, 
										mode = cv2.RETR_EXTERNAL, 
										method = cv2.CHAIN_APPROX_NONE) 
		return contours 

	def clean_plate(self, plate): 
		
		gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) 
		thresh = cv2.adaptiveThreshold(gray, 
									255, 
									cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
									cv2.THRESH_BINARY, 
									11, 2) 
		
		contours, _ = cv2.findContours(thresh.copy(), 
										cv2.RETR_EXTERNAL, 
										cv2.CHAIN_APPROX_NONE) 

		if contours: 
			areas = [cv2.contourArea(c) for c in contours] 
			
			# index of the largest contour in the area 

			# max_index = np.argmax(areas)
			max_cntArea = areas[np.argmax(areas)] 
			max_cnt = contours[np.argmax(areas)] 
			
			# (x, y) are the top left co-ordinates of the rectangle and w and h denotes the width and height respectively of the rectangle.
			x, y, w, h = cv2.boundingRect(max_cnt) 
			rect = cv2.minAreaRect(max_cnt) 
			if not self.ratioCheck(max_cntArea, plate.shape[1], 
												plate.shape[0]): 
				return plate, False, None
			
			return plate, True, [x, y, w, h] 
		
		else: 
			return plate, False, None
	
	def check_plate(self, input_img, contour): 
		
		min_rect = cv2.minAreaRect(contour) 
		
		if self.validateRatio(min_rect): 
			x, y, w, h = cv2.boundingRect(contour) 
			after_validation_img = input_img[y:y + h, x:x + w] 
			after_clean_plate_img, plateFound, coordinates = self.clean_plate( 
														after_validation_img) 
			
			if plateFound: 
				characters_on_plate = self.find_characters_on_plate( 
											after_clean_plate_img) 
				
				if (characters_on_plate is not None and len(characters_on_plate) <= 8): 
					x1, y1, w1, h1 = coordinates 
					coordinates = x1 + x, y1 + y 
					after_check_plate_img = after_clean_plate_img 
					
					return after_check_plate_img, characters_on_plate, coordinates 
		
		return None, None, None
