import cv2
import numpy as np


def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array([[
        (100, height),
        (width - 100, height),
        (width // 2, int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def make_coordinates(image, line_params):
    slope, intercept = line_params
    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_lines(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue  # skip vertical lines

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lanes = []

    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        lanes.append(make_coordinates(image, left_avg))

    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        lanes.append(make_coordinates(image, right_avg))

    return np.array(lanes)


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=40,
        maxLineGap=150
    )

    averaged_lines = average_lines(frame, lines)

    line_image = np.zeros_like(frame)

    if averaged_lines is not None:
        for x1, y1, x2, y2 in averaged_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 8)

    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Camera not accessible")
        return

    print("Lane detection running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = detect_lanes(frame)
        cv2.imshow("Lane Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
