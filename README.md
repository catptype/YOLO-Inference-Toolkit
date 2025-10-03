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

## Installation & Dependencies

This toolkit is designed to be directly integrated into your projects.

### Step 1: Add the Toolkit to Your Project

Copy the [**`YoloModel.py`**](util/YoloModel.py) file into a utility folder within your own project. You can then import the classes directly.

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

### 1. Classification ([Example Notebook](example_classify.ipynb))

### 2. Object Detection ([Example Notebook](example_detect.ipynb))

### 3. Instance Segmentation ([Example Notebook](example_segment.ipynb))

### 4. Object Tracking with Detection ([Example Notebook](example_tracking.ipynb))

## API Overview

The toolkit uses an inheritance structure to maximize code reuse.

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