# YOLO Inference Toolkit

A high-level, object-oriented Python toolkit for simplifying inference with the **Ultralytics YOLO framework**. This toolkit is designed to work with **any YOLO model** compatible with the Ultralytics library, making inference tasks more structured, reusable, and developer-friendly.

This toolkit abstracts away the boilerplate code, providing a clean and predictable API for common tasks like classification, object detection, and tracking. It separates the complex model output from the simple, structured data you actually need for your application.

> **Note**: This toolkit was developed and tested against `ultralytics` version `8.3.119`. Future versions of the library may introduce breaking changes.

## Key Features

-   **Model Agnostic**: Works with any detection or classification model supported by the Ultralytics framework.
-   **Clean Object-Oriented Design**: Specialized, inheritable classes for each task.
-   **Predict-then-Decode Pattern**: A flexible architecture that separates running inference from parsing the results, improving efficiency.
-   **Simplified Data Output**: Methods that return simple Python lists and tuples (e.g., `[(box, score, class_name)]` for detection or `(class_name, score)` for classification), not complex result objects.
-   **Batch & Single Image Processing**: Consistent methods that handle both single images and batches of images for high-throughput applications.
-   **Specialized Task Capabilities**: Includes advanced, ready-to-use implementations for common tasks, such as stateful object tracking and object extraction utilities for detection.
-   **Well-Documented**: Clear docstrings, type hinting, and straightforward examples.

## Installation & Dependencies

This toolkit is designed to be directly integrated into your projects.

### Step 1: Add the Toolkit to Your Project

Copy the Python file containing the `YoloModel`, `YoloClassification`, and `YoloDetection` classes (e.g., `yolo_toolkit.py`) into a utility folder within your own project (e.g., `/utils` or `/lib`). You can then import the classes directly.

### Step 2: Install Required Dependencies

Ensure the following packages are installed in your Python environment using pip:

```bash
pip install numpy ultralytics supervision opencv-python
```
Specific versions tested:
- `numpy>=1.23.5,<2.0.0`
- `ultralytics`
- `supervision`
- `opencv-python`

#### **Important Note for GPU Users**

For GPU-accelerated performance, you **must install PyTorch with CUDA support manually** before installing the packages above. The standard `ultralytics` installation will default to a CPU-only version of PyTorch.

1.  Go to the official PyTorch website: **[Link](https://pytorch.org/get-started/locally/)**
2.  Select the appropriate options for your system (e.g., Stable, Windows/Linux, Pip, Python, your CUDA version).
3.  Run the generated command. It will look something like this:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Usage

The toolkit provides a simple and consistent API across different tasks.

### 1. Classification ([Example](example_classify.ipynb))

#### Single Image Inference
This example shows how to classify a single image and display the result.

```python
import cv2
import matplotlib.pyplot as plt
from your_utils_folder.yolo_toolkit import YoloClassification # Import from where you placed the file

# 1. Initialize the classifier
yolo_model = YoloClassification("models/yolo11n-cls.pt")

# 2. Load and prepare the image
image = cv2.imread("images/classify/goldfish.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3. Predict and decode the result
results = yolo_model.predict(image)
cls_name, score = yolo_model.decode_top1(results)

# 4. Display the result
plt.imshow(image)
plt.title(f"Class: {cls_name} | Conf: {score:.3f}")
plt.axis('off')
plt.show()
```

#### Batch Image Inference
Process multiple images in a single call for higher throughput.

```python
import cv2
import matplotlib.pyplot as plt
from your_utils_folder.yolo_toolkit import YoloClassification

# 1. Initialize the classifier
yolo_model = YoloClassification("yolo11n-cls.pt")

# 2. Create a list of images
image_paths = ["images/classify/goldfish.jpg", "images/classify/goldfinch.jpg", "images/classify/ostrich.jpg"]
image_list = [cv2.imread(p) for p in image_paths]

# 3. Predict on the batch and decode results
results = yolo_model.predict(image_list)
decode_results = yolo_model.decode_top1(results)

# 4. Display the results for each image
for image, (cls_name, score) in zip(image_list, decode_results):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(f"Class: {cls_name} | Conf: {score:.3f}")
    plt.axis('off')
    plt.show()
```

### 2. Object Detection ([Example](example_detect.ipynb))

#### Single Image Inference
This example demonstrates detecting objects in one image, drawing bounding boxes, and extracting each detected object.

```python
import cv2
import matplotlib.pyplot as plt
from your_utils_folder.yolo_toolkit import YoloDetection # Import from where you placed the file

# 1. Initialize the detector
yolo_model = YoloDetection("yolo11n.pt")

# 2. Load and prepare the image
image = cv2.imread("images/detect/multiple1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_with_bbox = image.copy()

plt.imshow(image)
plt.title("Input")
plt.axis('off')
plt.show()

# 3. Predict and decode
results = yolo_model.predict(image)
decoded_results = yolo_model.decode_results(results)

# 4. Process each detected object
for box, score, cls_name in decoded_results:
    # Draw bounding box on the image copy
    pt1 = (box[0], box[1])
    pt2 = (box[2], box[3])
    cv2.rectangle(image_with_bbox, pt1, pt2, (255, 255, 0), 3)
    
    # Extract and display the cropped object
    crop_image = yolo_model.extract_object(image, box)
    plt.imshow(crop_image)
    plt.title(f"Class: {cls_name} | Score: {score:.3f}")
    plt.axis('off')
    plt.show()

# 5. Display the final image with all bounding boxes
plt.imshow(image_with_bbox)
plt.title("All Detections")
plt.axis('off')
plt.show()    
```

#### Batch Image Inference
Process a list of images in a single, efficient batch call.

```python
import cv2
import matplotlib.pyplot as plt
from your_utils_folder.yolo_toolkit import YoloDetection

# 1. Initialize the detector
yolo_model = YoloDetection("yolo11n.pt")

# 2. Create a list of images
image_paths = ["images/detect/multiple1.png", "images/detect/airplane.jpg", "images/detect/bicycle.jpg"]
image_list = [cv2.imread(p) for p in image_paths]

# 3. Predict on the batch and decode
results = yolo_model.predict(image_list)
decode_results = yolo_model.decode_results(results)

# 4. Iterate through each image and its corresponding detections
for image, objects in zip(image_list, decode_results):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_bbox = image.copy()
    
    print("=" * 20)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

    # Process each detected object in the current image
    for box, score, cls_name in objects:
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        cv2.rectangle(image_with_bbox, pt1, pt2, (255, 255, 0), 3)
        
        # Crop and display the object
        crop_image = yolo_model.extract_object(image, box)
        plt.imshow(crop_image)
        plt.title(f"Class: {cls_name} | Score: {score:.3f}")
        plt.axis('off')
        plt.show()

    plt.imshow(image_with_bbox)
    plt.title("All Detections in Image")
    plt.axis('off')
    plt.show()    
```

## API Overview

### `YoloModel` (Base Class)
-   `warmup()`: Pre-runs the model to reduce initial latency.
-   `decode_speed()`: Extracts inference timing information.
-   `class_names` (property): Returns a dictionary of class names.
-   `training_imgsz` (property): Returns the image size the model was trained on.

### `YoloClassification(YoloModel)`
-   `predict()`: Runs inference. Returns raw `ultralytics.Results`.
-   `decode_top1()`: Parses results to get the top prediction.
-   `decode_top5()`: Parses results to get the top 5 predictions.

### `YoloDetection(YoloModel)`
-   `predict()`: Runs stateless detection. Returns raw `ultralytics.Results`.
-   `tracking()`: Runs stateful tracking on a single frame. Returns a `supervision.Detections` object with tracker IDs.
-   `decode_results()`: Decodes raw `ultralytics.Results` into a simple list of detections.
-   `decode_detections()`: Decodes `supervision.Detections` (from tracking) into a simple list with tracker IDs.
-   `extract_object()`: A utility to crop an object from an image using its bounding box.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.