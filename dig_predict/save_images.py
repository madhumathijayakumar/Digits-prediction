import mysql.connector
from PIL import Image
import cv2
import pytesseract
import os
import requests
import numpy as np
import time


from cropper_validations import find_outermost_rectangle, find_fourth_point_v2, extrapolate_for_direction_right, \
    get_markers_direction, extrapolate_for_direction_bottom, extrapolate_for_direction_left, \
    extrapolate_for_direction_top, is_skewed, dimension_checker
from fix_tilted_image import correct_image_tilt


def startExecution(response, new_session_dir, image_download_domain):
    print("\n" * 2)
    print("startExecution ...", response)

    s3_path = response.get("s3_path")
    file_name = response.get("image_name")
    new_session_image_store = new_session_dir

    if not os.path.exists(new_session_image_store):
        os.makedirs(new_session_image_store, exist_ok=True)

    image_url = f"{image_download_domain}/{s3_path}"
    fetch_response = requests.get(image_url)
    image_file_path = os.path.join(new_session_image_store, file_name)
    print(image_file_path)


    if fetch_response.status_code == 200:
        print(fetch_response)
        with open(image_file_path, "wb") as f:
            print("open")
            f.write(fetch_response.content)
            print("image downloaded successfully ")

        # initial_cropped_image_path = os.path.join(new_session_dir, file_name)
        initial_cropped_image_path = "/home/madhu/cropped_images/20240606193111_pdf_page_39.png"
        crop_image_based_on_markers_v6(image_file_path, initial_cropped_image_path)

        # Secondary crop to extract the student's marks area
        formatted_file_name = file_name.replace('.png', '', 1)
        secondary_cropped_image_path = os.path.join(os.path.dirname(new_session_dir), f"{formatted_file_name}_marks.png")
        crop_marks_area_with_coordinates(initial_cropped_image_path, secondary_cropped_image_path)
        recognize_marks( secondary_cropped_image_path)
    else:
        print(f"Failed to download the image. Status code: {fetch_response.status_code}")

# Example usage:
response = {
    "s3_path": 'testing/PART-C/20240606193111/pdf_page_39_1718221675923.png',
    "image_name": "20240606193111_pdf_page_39.png"
}
new_session_dir = "/home/madhu/cropped_images"
image_download_domain = "https://docs.exampaper.vidh.ai"
start_time = time.time()  # Setting the start time
print(f"Execution time: {time.time() - start_time} seconds")


def get_s3_path_from_database(image_name):
    """ Function to retrieve s3_path from MySQL database using image_name """
    try:
        conn = mysql.connector.connect(
            host="34.131.182.12",
            user="rootuser",
            password="1234",
            database="vidhai_ms_solutions_dev"
        )
        cursor = conn.cursor()
        sql = "SELECT s3_path FROM ocr_scanned_part_c_v1 WHERE image_name = %s"
        image_name = '20240606193111_pdf_page_39.png'
        cursor.execute(sql,image_name)
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        print(f"Error fetching s3_path from database: {e}")
        return None

count = 0

def crop_image_based_on_markers_v6(image_path, output_path):
    try:
        is_titlted = correct_image_tilt(image_path, image_path)
        global count
        error_code = None
        error_reason = None
        det_marker_cnt = None
        print("try with outer markers...")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image.")
            return

        # Make a copy of the original image to draw markers on
        marked_image = image.copy()

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply localized thresholding to get a binary image
        block_size = 153  # Larger block size for better local adaptation
        constant = 20  # Adjust constant for better thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size,
                                       constant)

        # imS = cv2.resize(thresh, (960, 540))
        # cv2.imshow("thresh", imS)
        # cv2.waitKey()
        # cv2.destroyAllWindows
        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours based on area (descending order)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Draw all contours on the marked image
        # cv2.drawContours(marked_image, contours, -1, (0, 0, 255), 2)  # Draw all contours in red
        # imS = cv2.resize(marked_image, (960, 540))
        # cv2.imshow("thresh", imS)
        # cv2.waitKey()
        # cv2.destroyAllWindows
        # List to hold the marker points
        markers = []

        expected_marker_size = (67, 65)
        min_area = 3100
        max_area = 4800

        print(min_area, max_area)
        # Iterate over the contours to find the largest square markers
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)  # Increased accuracy
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # print("area = ", area)
            # print(x,y,w,h)
            # print(aspect_ratio)

            if min_area <= area <= max_area and 0.7 <= aspect_ratio <= 1.2 and 60 <= w < 72 and 60 <= h < 78:
                ## suj check

                # print("area = ", area)
                # print(x,y,w,h)
                # print(aspect_ratio)
                # print("true area: ", area)
                marker = (x + w // 2, y + h // 2)
                markers.append(marker)
                # cv2.circle(marked_image, marker, 10, (0, 255, 0), -1)

                # if len(markers) == 4:
                #     break

        print("Detected markers:", markers)
        det_marker_cnt = len(markers)
        ref_markers = markers

        if len(markers) > 4:
            print("found marker is greater than 4")
            is_titlted = correct_image_tilt(image_path, image_path)
            print("is_titlted = ", is_titlted)
            print("count = ", count)
            # if is_titlted and count < 1:
            #     count += 1
            #     st,error_code, error_reason, det_marker_cnt = crop_image_based_on_markers_v6(image_path, output_path)
            #     return st, error_code, error_reason, det_marker_cnt
            outermost_rectangle = find_outermost_rectangle(markers)
            print("outermost_rectangle = ", outermost_rectangle)
            markers = outermost_rectangle


        elif len(markers) == 4:
            print("found markers is 4")
            markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x

            # Manually assign top-left, top-right, bottom-left, bottom-right
            top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
            bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
            markers = [top_left, top_right, bottom_left, bottom_right]
            print("sorted marker = ", markers)


        elif len(markers) == 3:
            print("only 3 markers is being found")
            fourth_point = find_fourth_point_v2(markers)
            if fourth_point:
                markers.append(fourth_point)
                # markers = find_fourth_point_v2(markers)
                markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                print("sorted = ", markers)
                # Manually assign top-left, top-right, bottom-left, bottom-right
                top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                markers = [top_left, top_right, bottom_left, bottom_right]
                print("sorted marker = ", markers)
            else:
                print("4th point not found..")
                error_code = 103
                error_reason = "fourth point prediction failed"

        elif len(markers) == 2:

            print("tilted good")
            print("is_titlted = ", is_titlted)
            print("count = ", count)
            # if is_titlted and count < 1:
            #     count += 1
            #     st,error_code, error_reason, det_marker_cnt = crop_image_based_on_markers_v6(image_path, output_path)

            with Image.open(image_path) as img:
                image_w, image_h = img.size

            print(image_w, image_h)
            direction = get_markers_direction(image_w, image_h, markers)
            print("direction:", direction)
            if not direction:
                print("Invalid direction!!!")

            elif direction and direction == "right":
                markers = extrapolate_for_direction_right(markers)
                print(markers)
                if len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "failed right"

            elif direction and direction == "bottom":
                print("bottom")
                markers = extrapolate_for_direction_bottom(markers)
                print(markers)
                if len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "failed bottom"

            elif direction and direction == "left":
                print("left")
                markers = extrapolate_for_direction_left(markers)
                print(markers)
                if markers and len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "left"
                    return False, error_code, error_reason, det_marker_cnt
            # elif direction and direction == "diag-back":
            #     print("diag-back")
            #     markers = extrapolate_for_direction_diag_back (markers)
            #     print (markers)
            #     if len(markers) == 4:
            #         markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
            #         print("sorted = ", markers)
            #         # Manually assign top-left, top-right, bottom-left, bottom-right
            #         top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
            #         bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
            #         markers = [top_left, top_right, bottom_left, bottom_right]
            #         print("sorted marker = ", markers)
            #     else:
            #         print("marker count mismatch...")
            #         error_code = 103
            #         error_reason = "two point prediction failed"

            elif direction and direction == "top":
                print("diag-back")
                markers = extrapolate_for_direction_top(markers)
                print(markers)
                if len(markers) == 4:
                    markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
                    print("sorted = ", markers)
                    # Manually assign top-left, top-right, bottom-left, bottom-right
                    top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
                    bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
                    markers = [top_left, top_right, bottom_left, bottom_right]
                    print("sorted marker = ", markers)
                else:
                    print("marker count mismatch...")
                    error_code = 103
                    error_reason = "failed top"

            # elif direction and direction == "diag-front":
            #     print("diag-back")
            #     markers = extrapolate_for_direction_diag_front (markers)
            #     print (markers)
            #     if len(markers) == 4:
            #         markers = sorted(markers, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
            #         print("sorted = ", markers)
            #         # Manually assign top-left, top-right, bottom-left, bottom-right
            #         top_left, top_right = sorted(markers[:2], key=lambda p: p[0])
            #         bottom_left, bottom_right = sorted(markers[2:], key=lambda p: p[0])
            #         markers = [top_left, top_right, bottom_left, bottom_right]
            #         print("sorted marker = ", markers)
            #     else:
            #         print("marker count mismatch...")
            #         error_code = 103
            #         error_reason = "two point prediction failed"

            else:
                error_code = 103
                error_reason = direction
                return False, error_code, error_reason, det_marker_cnt

        else:
            print("crop failed...")
            print(f"no of markers detected is {len(markers)}")
            print("Trying with tilt correction...")
            # marking(marked_image, ref_markers)
            # is_titlted = correct_image_tilt(image_path, image_path)
            # print("is_titlted = ", is_titlted)
            # print("count = ", count)
            # if is_titlted and count < 1:
            #     count += 1
            #     st,error_code, error_reason, det_marker_cnt = crop_image_based_on_markers_v6(image_path, output_path)
            #     return st,error_code, error_reason, det_marker_cnt
            print(f"after tilt correction also failed with marker count: {len(markers)}")
            error_code = 103
            error_reason = "marker count issue"
            # det_marker_cnt = len(markers)
            return False, error_code, error_reason, det_marker_cnt

        print("markedr = ", markers)

        if markers:
            st, error_reason = crop_validation(markers)
            print(st)
            if st:
                crop_image(markers, image, marked_image, output_path)
                print("crop successful")
                return True, error_code, error_reason, det_marker_cnt
            else:
                print("crop failed...")
                # is_titlted = correct_image_tilt(image_path, image_path)
                # print("is_titlted = ", is_titlted)
                # print("count = ", count)
                # if is_titlted and count < 1:
                #     count += 1
                #     st,error_code, error_reason, det_marker_cnt = crop_image_based_on_markers_v6(image_path, output_path)
                #     return st,error_code, error_reason, det_marker_cnt
                # marking(marked_image, ref_markers)
                print(f"after tilt correction also failed with marker count: {len(markers)}")
                error_code = 103
                det_marker_cnt = len(markers)
                return False, error_code, error_reason, det_marker_cnt
        else:
            # marking(marked_image, ref_markers)
            print("marker len is not 4")
            print(f"after tilt correction also failed with marker count 0")
            error_code = 103
            error_reason = "marker count issue"
            det_marker_cnt = len(markers)
            return False, error_code, error_reason, det_marker_cnt

    except Exception as e:
        print(f"Error in cropping: {e}")
        error_code = 103
        return False, error_code, error_reason, det_marker_cnt

def crop_validation(markers):
    print("inside cropvalidation...")
    error_reason = None
    try:
        if markers and len(markers) == 4:
            st = is_skewed(markers)
            print("status = ", st)
            if st:
                st2 = dimension_checker(markers)
                if st2:
                    print("dimension check st2 = ", st2)
                    return st2, error_reason
                else:
                    error_reason = "failed in dimension check"
                    print("dimension check st2 = ", st2)
                    return st2, error_reason
            else:
                print("skewed image found...")
                error_reason = "failed in skew check"
                return None, error_reason
        else:
            print("marker length is not equals to 4")
            error_reason = "marker length is not four"
            return None, error_reason

    except Exception as e:
        print(f"Error in crop_validation: {e}")
        return None, error_reason



def crop_image(markers, image, marked_image, output_path):
    print("inside cropping image")

    print("Sorted markers:", markers)

    # Draw green circles on the markers and label them
    # labels = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
    # for i, marker in enumerate(markers):
    #     cv2.circle(marked_image, marker, 10, (0, 255, 0), -1)
    #     cv2.putText(marked_image, labels[i], (marker[0] + 10, marker[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the marked image with annotations
    # cv2.imwrite("s3_images/marked_v2.png", marked_image)

    # Define the points for perspective transform
    pts1 = np.float32([markers[0], markers[1], markers[2], markers[3]])
    width, height = image.shape[1], image.shape[0]
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    print("Points for perspective transform:\n", pts1, "\n", pts2)

    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (width, height))

    # Save the output image
    cv2.imwrite(output_path, result)
    return True

def crop_marks_area_with_coordinates(input_image_path, output_image_path):

    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Define cropping coordinates and size
    x, y, width, height = 2352, 272, 264, 165

    # Calculate end coordinates
    x2 = x + width
    y2 = y + height

    # Ensure coordinates are within image bounds
    if x < 0 or y < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print("Error: Cropping coordinates out of image bounds.")
        return

    # Crop the marks area
    cropped_marks_area = image[y:y2, x:x2]

    # Check if cropped image is empty
    if cropped_marks_area is None or cropped_marks_area.size == 0:
        print("Error: Cropped marks area is empty.")
        return

    # Save the cropped image
    cv2.imwrite(output_image_path, cropped_marks_area)
    print(f"Marks area image saved at: {output_image_path}")

    # Recognize marks in the cropped image (assuming a separate function exists)
    recognize_marks(cropped_marks_area)  # Pass the cropped image directly
def recognize_marks(secondary_cropped_image_path):  # Function now takes a list of image paths

    try:
        image = cv2.imread(secondary_cropped_image_path)

        # Check if image is loaded successfully
        if image is None:
            raise FileNotFoundError(f"Could not find or open image: {secondary_cropped_image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve OCR accuracy
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Optional: Dilate the image to make text more pronounced
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(thresh, kernel, iterations=1)

        # Perform OCR using Tesseract with appropriate configuration
        custom_config = r'--oem 3 --psm 6'  # OEM 3 for default, PSM 6 for single block of text
        text = pytesseract.image_to_string(dilate, config=custom_config)

        # Print the recognized marks
        print(f"Image: {secondary_cropped_image_path}, Recognized Marks: {text.strip()}")

    except Exception as e:
        print(f"An error occurred for image {secondary_cropped_image_path}: {e}")


image_paths = [
     "/home/madhu/cropped_images/20240606192827_pdf_page_1_marks.png"
]

for image_path in image_paths:
    recognize_marks(image_path)

if __name__ == "__main__":
    # train_mnist_cnn_model()
    startExecution(response, new_session_dir, image_download_domain)
    # load_and_preprocess_image(secondary_cropped_image_path)
