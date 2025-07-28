from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
import tempfile

app = Flask(__name__)

def save_and_return_image(image):
    temp_filename = tempfile.mktemp(suffix='.png')
    cv2.imwrite(temp_filename, image)
    return send_file(temp_filename, mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    if request.method == 'POST':
        operation = request.form.get('operation')
        sub_operation = request.form.get('sub_operation')
        image_choice1 = request.form.get('image_choice1')

        img_1_path = image_choice1
        img_1 = cv2.imread(img_1_path)

        if img_1 is None:
            return "Image not found. Ensure the chosen image is in the 'static' directory."

        processed_image = None

        try:
            if operation == 'sobel':
                gray_img = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
                grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                processed_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            elif operation == 'hist':
                gray_img = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                equ = cv2.equalizeHist(gray_img)
                processed_image = np.hstack((gray_img, equ))

            elif operation == 'erode':
                kernel = np.ones((5, 5), np.uint8)
                processed_image = cv2.erode(img_1, kernel, iterations=1)

            elif operation == 'dilate':
                kernel = np.ones((5, 5), np.uint8)
                processed_image = cv2.dilate(img_1, kernel, iterations=1)

            elif operation == 'change_hue':
                hsv_img = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + 30) % 180  # Adding 30 to hue channel
                processed_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

            elif operation == 'grabCut':
                mask = np.zeros(img_1.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                rect = (50, 50, img_1.shape[1] - 100, img_1.shape[0] - 100)
                cv2.grabCut(img_1, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                processed_image = img_1 * mask2[:, :, np.newaxis]

            elif operation == 'findContours':
                gray_img = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_img, 127, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_1, contours, -1, (0, 255, 0), 3)
                processed_image = img_1

            elif operation == 'blur':
                if sub_operation == 'median':
                    processed_image = cv2.medianBlur(img_1, 5)
                elif sub_operation == 'gaussian':
                    processed_image = cv2.GaussianBlur(img_1, (5, 5), 0)
                elif sub_operation == 'blur':
                    processed_image = cv2.blur(img_1, (5, 5))
                elif sub_operation == 'boxFilter':
                    processed_image = cv2.boxFilter(img_1, -1, (5, 5))

            elif operation == 'Canny':
                processed_image = cv2.Canny(img_1, 100, 200)

            elif operation == 'cvtColor':
                processed_image = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

            elif operation == 'cornerHarris':
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                dst = cv2.cornerHarris(gray, 2, 5, 0.07)
                dst = cv2.dilate(dst, None)
                img_1[dst > 0.01 * dst.max()] = [0, 0, 255]
                processed_image = img_1

            if processed_image is not None:
                return save_and_return_image(processed_image)
            else:
                return "Invalid operation or sub-operation"

        except Exception as e:
            return f"An error occurred: {str(e)}"

    return "Invalid request"

if __name__ == '__main__':
    app.run(debug=True)
