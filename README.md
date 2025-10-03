# YOLO Inference Toolkit

A high-level, object-oriented Python toolkit for simplifying inference with the **Ultralytics YOLO framework**. This toolkit is designed to work with **any YOLO model** compatible with the Ultralytics library, making tasks like classification, detection, segmentation, and pose estimation more structured, reusable, and developer-friendly.

This toolkit abstracts away the boilerplate code, providing a clean and predictable API. It separates the complex model output from the simple, structured data you actually need for your application.

> **Note**: This toolkit was developed and tested against `ultralytics` version `8.3.119`. Future versions of the library may introduce breaking changes.

## Key Features

-   **Model Agnostic**: Works with any classification, detection, segmentation, or pose model supported by the Ultralytics framework.
-   **Clean Object-Oriented Design**: A logical, inheritable class structure (`YoloModel` -> `YoloObjectBase` -> `YoloDetection`).
-   **Predict-then-Decode Pattern**: A flexible architecture that separates running inference from parsing the results, improving efficiency.
-   **Simplified Data Output**: Methods that return simple Python lists and tuples (e.g., `[(box, score, class_name)]` for detection or `(class_name, score)` for classification), not complex result objects.
-   **Batch & Single Image Processing**: Consistent methods that handle both single images and batches of images for high-throughput applications.
-   **Specialized Task Capabilities**: Includes advanced, ready-to-use implementations for common tasks, such as stateful object tracking and utilities for object extraction.
-   **Well-Documented**: Clear docstrings, type hinting, and straightforward examples.

## Getting Started

This toolkit is simply designed to be directly integrated into your projects.

### Step 1: Install Dependencies

First, ensure the required packages are installed in your Python environment.

```bash
pip install numpy ultralytics supervision opencv-python
```

> **Note on NumPy**: This project was tested with `numpy<2.0.0`. It is recommended to ensure your version is compatible.

#### **For GPU Acceleration (Recommended)**

The `ultralytics` library requires PyTorch. For GPU acceleration, you **must install a CUDA-enabled version of PyTorch manually *before* installing the other packages**. The official PyTorch website provides the exact command for your system.

1.  Go to the official PyTorch website: **[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)**
2.  Select the appropriate options for your system and run the generated command.

### Step 2: Integrate the Toolkit

Simply copy the [**`YoloModel.py`**](util/YoloModel.py) file into a utility folder in your project. You can then import the classes directly into your scripts. It's that easy!

## Usage Examples

Below are quick-start examples for each primary task. For more detailed, runnable code, please see the linked Jupyter Notebooks.

### Classification
➡️ **Full Example Notebook:** [**`example_classify.ipynb`**](example_classify.ipynb)

This example shows how to classify a single image and decode the top-1 result.

```python
import cv2
from util.YoloModel import YoloClassification

# Initialize and predict
yolo_cls = YoloClassification("models/yolo11n-cls.pt")
image = cv2.imread("images/classify/goldfish.jpg")
results = yolo_cls.predict(image)

# Decode and use the results
class_name, score = yolo_cls.decode_top1(results)
print(f"Detected: {class_name} with confidence {score:.2f}")
```

### Object Detection & Tracking
➡️ **Full Example Notebook:** [**`example_detect.ipynb`**](example_detect.ipynb)

This example demonstrates detecting objects, decoding the results, and extracting a detected object from the image.

```python
import cv2
from util.YoloModel import YoloDetection

# Initialize and predict
yolo_det = YoloDetection("models/yolo11n.pt")
image = cv2.imread("images/detect/multiple1.png")
results = yolo_det.predict(image)

# Decode and process each detected object
decoded_results = yolo_det.decode_results(results)
for box, score, class_name in decoded_results:
    print(f"Found {class_name} at {box}")
    # Utility to crop the detected object from the frame
    cropped_object = yolo_det.extract_object(image, box)
```

For tracking object, simply switch from `predict()` to `tracking()` and from `decode_results()` to `decode_detections()` methods to get persistent object IDs across images.

```python
import cv2
from util.YoloModel import YoloDetection

# Initialize for tracking
yolo_det = YoloDetection("models/yolo11n.pt")
image = cv2.imread("images/detect/multiple1.png")
results = yolo_det.tracking(frame)

# Decode results to get the tracker_id
decoded_results = yolo_det.decode_detections(results)
for tracker_id, box, score, class_name in decoded_results:
    print(f"Object ID: {tracker_id} is a {class_name} at {box}")
```

### Instance Segmentation
➡️ **Full Example Notebook:** [**`example_segment.ipynb`**](example_segment.ipynb)

This example shows how to get segmentation masks and use a utility to isolate and extract a segmented object.

```python
import cv2
from util.YoloModel import YoloSegmentation

# Initialize and predict
yolo_seg = YoloSegmentation("models/yolo11n-seg.pt")
image = cv2.imread("images/detect/multiple1.png")
results = yolo_seg.predict(image)

# Decode and process each segmented object
decoded_results = yolo_seg.decode_results(results)
for box, mask, score, class_name in decoded_results:
    # Utility to apply the mask and crop the object
    segmented_object = yolo_seg.segment_object(image, box, mask)
```

### Pose Estimation
➡️ **Full Example Notebook:** [**`example_pose.ipynb`**](example_pose.ipynb)

This example shows how to detect human poses and extract keypoint data for each instance.

```python
import cv2
from util.YoloModel import YoloPose

yolo_pose = YoloPose("models/yolo11n-pose.pt")
image = cv2.imread("images/pose/person1.jpg")
image_with_keypoints = image.copy()
results = yolo_pose.predict(image)

decoded_results = yolo_pose.decode_results(results)
for box, keypoints, score, class_name in decoded_results:
    # Draw each keypoint as a circle on the image
    for (x, y) in keypoints:
        cv2.circle(image_with_keypoints, (x, y), 5, (0, 255, 0), -1)
```

## Class Reference

The toolkit uses an inheritance structure to maximize code reuse and clarity.

### `YoloModel` (Base Class)
-   The top-level class for all models.

### `YoloClassification(YoloModel)`
-   **`predict()`**: Runs inference.
-   **`decode_top1()`**: Decodes the top-1 prediction `(class_name, score)`.
-   **`decode_top5()`**: Decodes the top-5 predictions `[(class_name, score)]`.

### `YoloObjectBase(YoloModel)`
-   An intermediate class for object-based tasks (detection, segmentation, pose).
-   **`predict()`**: Runs stateless inference for detection-based tasks.
-   **`tracking()`**: Runs stateful, frame-by-frame tracking.
-   **`extract_object()`**: Utility to crop a rectangular region from an image.

### `YoloDetection(YoloObjectBase)`
-   **`decode_results()`**: Decodes raw results into `[(box, score, class_name)]`.
-   **`decode_detections()`**: Decodes tracked results into `[(tracker_id, box, score, class_name)]`.

### `YoloSegmentation(YoloObjectBase)`
-   **`decode_results()`**: Decodes raw results into `[(box, polygon_mask, score, class_name)]`.
-   **`decode_detections()`**: Decodes tracked results into `[(tracker_id, box, binary_mask, score, class_name)]`.
-   **`segment_object()`**: Utility to extract a segmented object using its mask.

### `YoloPose(YoloObjectBase)`
-   **`decode_results()`**: Decodes raw results into `[(box, keypoints, score, class_name)]`.
-   **`decode_detections()`**: Decodes tracked results into `[(tracker_id, box, keypoints, score, class_name)]`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.