import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def correct_image_tilt(image_path, output_path):
    try:
        # Load the image using OpenCV
        image_cv = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Use GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges for better angle detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Use HoughLinesP to detect line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        # cv2.imshow("lines", lines)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        # print("lines = ", lines)
        
        if lines is not None:
            # Calculate the angles of the detected line segments
            angles = []

            for line in lines:
                for x1, y1, x2, y2 in line:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    angles.append(angle)

            # Calculate the median angle
            median_angle = np.median(angles)

            # Rotate the image to correct the orientation
            (h, w) = image_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            # corrected_image = cv2.warpAffine(image_cv, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            corrected_image = cv2.warpAffine(image_cv, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            # Convert corrected image back to PIL format and save it
            corrected_image_pil = Image.fromarray(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
            corrected_image_pil.save(output_path)
            # corrected_image_pil.save("s3_images/tilted.png")
            
            # Display the original and corrected images side by side for comparison
            # plt.figure(figsize=(10, 5))

            # plt.subplot(1, 2, 1)
            # plt.title("Original Image")
            # plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

            # plt.subplot(1, 2, 2)
            # plt.title("Corrected Image")
            # plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))

            # plt.show()
            
            return True
        else:
            print("No lines detected. Cannot correct tilt.")
            return False
    except Exception as e:
        print("Error during tilt correction: ", e)
        return False

# Usage
# image_path = "s3_images/pdf_page_9_1717877522034.png"
# output_path = "s3_images/tilt_corrected.png"
# correct_image_tilt(image_path, output_path)
