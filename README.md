# Detect-the-lines-using-Hough-Transform-Record-HOUGH-TRANSFORM-
# Lane Detection

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

**Important Instruction:**
👉 Write code **ONLY in places marked as `# Your Code Here`**
👉 Do NOT modify any other part of the code

---

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

##  Algorithm & Explanation

---

###  Step 1: Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

###  Step 2: Read the Image

```python
# Read the image using OpenCV
img = cv2.imread('road.jpg')

# OpenCV loads images as BGR, so convert to RGB for correct display in matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Image shape:", img.shape)
```

---

###  Step 3: Convert to Grayscale

```python
# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

---

###  Step 4: Display Images

```python
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')

plt.show()
```

---

###  Step 5: Thresholding

```python
threshold = 150
_, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6,4))
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded")
plt.axis('off')
plt.show()
```

---

###  Step 6: Region of Interest (ROI)

```python
# ROI masking (do not modify)
height, width = thresh.shape
polygon = np.array([[
    (0, height),
    (int(width * 0.45), int(height * 0.6)),
    (int(width * 0.55), int(height * 0.6)),
    (width, height)
]], dtype=np.int32)

mask = np.zeros_like(thresh)
cv2.fillPoly(mask, polygon, 255)
roi = cv2.bitwise_and(thresh, mask)

plt.figure(figsize=(6,4))
plt.imshow(roi, cmap='gray')
plt.title("ROI Masked")
plt.axis('off')
plt.show()
```

---

### Step 7: Edge Detection (Canny)

```python
# Perform Edge Detection
edges = cv2.Canny(roi, 50, 150)

plt.figure(figsize=(6,4))
plt.imshow(edges, cmap='gray')
plt.title("Edges (Canny)")
plt.axis('off')
plt.show()
```

---

###  Step 8: Gaussian Blur

```python
# Apply Gaussian Blur to smooth the edge image
blurred = cv2.GaussianBlur(edges, (5, 5), 0)

plt.figure(figsize=(6,4))
plt.imshow(blurred, cmap='gray')
plt.title("Smoothed")
plt.axis('off')
plt.show()
```

---

###  Step 9: Hough Transform

```python
# Detect lines using Hough Transform (probabilistic version)
lines = cv2.HoughLinesP(
    blurred,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=40,
    maxLineGap=5
)

# Draw the raw detected lines on a blank image
line_img = np.zeros_like(img)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

plt.figure(figsize=(6,4))
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines")
plt.axis('off')
plt.show()
```

---

### Step 10: Lane Detection Logic

```python
# Lane detection logic (do not modify)
def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_lanes(image, lines):
    left_fit, right_fit = [], []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2:
            continue
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if slope < -0.3:
            left_fit.append((slope, intercept))
        elif slope > 0.3:
            right_fit.append((slope, intercept))

    lanes = []
    if left_fit:
        lanes.append(make_coordinates(image, np.average(left_fit, axis=0)))
    if right_fit:
        lanes.append(make_coordinates(image, np.average(right_fit, axis=0)))
    return np.array(lanes)

lanes = average_lanes(img, lines)

lane_img = np.zeros_like(img)
if lanes is not None:
    for x1, y1, x2, y2 in lanes:
        cv2.line(lane_img, (x1, y1), (x2, y2), (0, 255, 0), 10)

# Combine the lane lines with the original image
final = cv2.addWeighted(img, 0.8, lane_img, 1, 0)

plt.figure(figsize=(10,5))
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title("Final Lane Detection")
plt.axis('off')
plt.show()
```

---

##  Expected Output

* Original image  
<img width="342" height="526" alt="image" src="https://github.com/user-attachments/assets/54341960-5c1e-4916-a41c-82f361a3f733" />

* Grayscale image
<img width="347" height="523" alt="image" src="https://github.com/user-attachments/assets/e224a6ca-f807-48a5-8a1d-27592ef3ffa0" />

* Thresholded image
<img width="281" height="430" alt="image" src="https://github.com/user-attachments/assets/43025540-a2e9-46ed-a49e-99fdd3ee70f5" />

* ROI masked image
<img width="282" height="431" alt="image" src="https://github.com/user-attachments/assets/ea0cc561-fb0b-49e2-a167-bc6099e03155" />

* Edge detected image
<img width="280" height="432" alt="image" src="https://github.com/user-attachments/assets/b3384952-26ea-4453-9882-0797b510fc67" />

* Smoothed image
<img width="277" height="427" alt="image" src="https://github.com/user-attachments/assets/a804105b-ee83-4dc0-b832-c0253233652b" />

* Detected lines
<img width="280" height="435" alt="image" src="https://github.com/user-attachments/assets/a05246dc-d7cf-4a03-b472-c27e22e57fb1" />

* Final lane detection output
<img width="346" height="523" alt="image" src="https://github.com/user-attachments/assets/79da634a-7313-487e-b8a1-1be2c0da6928" />


---

##  Instructions

* Fill ONLY in `# Your Code Here` sections
* Do NOT change existing code
* Run step-by-step
* Verify outputs

---

## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---

##  Developed By

* **Name: JINITH KUMAR V**
* **Register No: 212225040157**
