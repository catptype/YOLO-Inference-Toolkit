# ==================================
#         Import Dependencies
# ==================================
import cv2
import numpy as np
import supervision as sv
from typing import Union, List, Any, Tuple, Dict
from ultralytics import YOLO
from ultralytics.engine.results import Results
from supervision.detection.core import Detections

# ==================================
#      YoloModel (Base Class)
# ==================================
class YoloModel:
    """
    A base class for wrapping Ultralytics YOLO models.

    This class provides a common interface for loading models and decoding
    standard information like inference speed and class names.
    """
    def __init__(self, model_path: str, task: str) -> None:
        """
        Initializes the YoloModel.

        Args:
            model_path (str): The path to the YOLO model file (e.g., 'yolov8n.pt').
            task (str): The task for the model, e.g., 'detect', 'classify'.
        """
        self.model = YOLO(model_path, task=task)

    def warmup(self, imgsz: int = 640) -> None:
        """
        Warms up the model with a dummy input to reduce latency on the first real inference.

        Args:
            imgsz (int): The image size to use for the dummy input. Should be
                         representative of the actual input size.
        """
        print("Warming up the model...")
        dummy_input = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy_input, verbose=False)
        print("Warm-up complete.")

    def decode_image_size(self, results: Union[Results, List[Results]]) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Decodes the original image size(s) from a Results object or a list of them.

        Args:
            results: A single Results object or a list of them.

        Returns:
            The original shape (height, width) of the input image(s).
        """
        if isinstance(results, list):
            return [r.orig_shape for r in results]
        else:
            return results.orig_shape

    def decode_speed(self, results: Union[Results, List[Results]]) -> Dict:
        """
        Decodes the speed/latency info. For a batch, it returns the single speed dict for the entire batch.

        Args:
            results: A single Results object or a list of them.

        Returns:
            A dictionary containing preprocess, inference, and postprocess times in ms.
        """
        if isinstance(results, list):
            # For a batch, the speed is the same for all results, so take the first.
            return results[0].speed if results else {}
        else:
            return results.speed

    @property
    def class_names(self) -> Dict[int, str]:
        """Returns the dictionary mapping class indices to class names."""
        return self.model.names
    
    @property
    def training_imgsz(self) -> int:
        """Returns the image size the model was trained with."""
        return self.model.args['imgsz']
    
# ==================================
#    YoloClassification (Subclass)
# ==================================
class YoloClassification(YoloModel):
    """
    A specialized class for YOLO classification tasks.
    """
    def __init__(self, model_path: str, warmup: bool = True) -> None:
        """
        Initializes the classification model.

        Args:
            model_path (str): Path to the classification model file.
            warmup (bool): If True, runs a dummy inference to warm up the model.
        """
        super().__init__(model_path, task='classify')
        if warmup:
            self.warmup(imgsz=self.training_imgsz or 224)

    def predict(self, source: Union[np.ndarray, List[np.ndarray]], **kwargs: Any) -> Union[Results, List[Results]]:
        """
        Runs classification inference and returns the raw Results object(s).
        Handles both single and batch inputs.

        Args:
            source: A single image (np.ndarray) or a list of images.
            **kwargs: Additional arguments for the ultralytics predict method.

        Returns:
            A single Results object or a list of them, corresponding to the input.
        """
        is_batch = isinstance(source, list)
        results = self.model.predict(source, verbose=False, **kwargs)
        return results if is_batch else results[0]

    def decode_top1(self, results: Union[Results, List[Results]]) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """
        Decodes the top-1 class name and score from classification results.

        Args:
            results: A single Results object or a list of them.

        Returns:
            For a single result: A (class_name, score) tuple.
            For a list of results: A list of (class_name, score) tuples.
        """
        if isinstance(results, list):
            return [self._decode_single_top1(res) for res in results]
        else:
            return self._decode_single_top1(results)

    def decode_top5(self, results: Union[Results, List[Results]]) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """
        Decodes the top-5 class names and scores from classification results.

        Args:
            results: A single Results object or a list of them.

        Returns:
            For a single result: A list of the top 5 (class_name, score) tuples.
            For a list of results: A list containing a list of top 5 tuples for each image.
        """
        if isinstance(results, list):
            return [self._decode_single_top5(res) for res in results]
        else:
            return self._decode_single_top5(results)

    def _decode_single_top1(self, result: Results) -> Tuple[str, float]:
        """Helper to decode a single Results object for the top-1 prediction."""
        idx = result.probs.top1
        name = self.class_names[idx]
        score = result.probs.top1conf.item()
        return (name, score)

    def _decode_single_top5(self, result: Results) -> List[Tuple[str, float]]:
        """Helper to decode a single Results object for the top-5 predictions."""
        indices = result.probs.top5
        names = [self.class_names[i] for i in indices]
        scores = result.probs.top5conf.tolist()
        return list(zip(names, scores))

# ==================================
# YoloObjectBase (Intermediate Base Class)
# ==================================
class YoloObjectBase(YoloModel):
    """
    An intermediate base class for object-based tasks (detection, segmentation, pose).
    Contains shared logic for prediction, tracking, and basic object extraction.
    """
    def __init__(self, model_path: str, task: str, warmup: bool = True) -> None:
        """
        Initializes the base model for object-based tasks.

        Args:
            model_path (str): Path to the model file.
            task (str): The specific task for the model, e.g., 'detect', 'segment'.
            warmup (bool): If True, runs a dummy inference to warm up the model.
        """
        super().__init__(model_path, task=task)
        if warmup:
            self.warmup(imgsz=self.training_imgsz or 640)
        self.tracker = sv.ByteTrack()

    def _set_default_kwargs(self, kwargs: Dict) -> Dict:
        """Helper to set default prediction arguments to avoid code repetition."""
        kwargs.setdefault('conf', 0.5); kwargs.setdefault('iou', 0.5)
        kwargs.setdefault('max_det', 50); kwargs.setdefault('agnostic_nms', True)
        return kwargs

    def predict(self, source: Union[np.ndarray, List[np.ndarray]], **kwargs: Any) -> Union[Results, List[Results]]:
        """
        Runs stateless inference. Handles both single and batch sources.

        Args:
            source: A single image (np.ndarray) or a list of images.
            **kwargs: Additional ultralytics predict arguments. If not provided,
                defaults are used (e.g., conf=0.5, iou=0.5).

        Returns:
            A single ultralytics.Results object or a list of them.
        """
        is_batch = isinstance(source, list)
        kwargs = self._set_default_kwargs(kwargs)
        results = self.model.predict(source, verbose=False, **kwargs)
        return results if is_batch else results[0]
    
    def tracking(self, source: np.ndarray, **kwargs: Any) -> Detections:
        """
        Performs prediction on a single frame and updates the stateful tracker.
        This method does NOT support batch processing.

        Args:
            source (np.ndarray): A single image or video frame.
            **kwargs: Additional arguments for the predict call. If not provided,
                defaults are used (e.g., conf=0.5, iou=0.5).

        Returns:
            A supervision.Detections object with tracker IDs assigned.
        
        Raises:
            ValueError: If the source is a list (batch input).
        """
        if isinstance(source, list):
            raise ValueError("The 'tracking' method does not support batch inputs.")
        kwargs = self._set_default_kwargs(kwargs)
        results: Results = self.model.predict(source, verbose=False, **kwargs)[0]
        detections = Detections.from_ultralytics(results)
        return self.tracker.update_with_detections(detections)
    
    def extract_object(self, image: np.ndarray, box_xyxy: Tuple[int, int, int, int], offset: int = 0) -> np.ndarray:
        """
        Crops a rectangular area from an image using a bounding box.

        Args:
            image (np.ndarray): The source image.
            box_xyxy (Tuple[int, int, int, int]): A tuple of (x1, y1, x2, y2) coordinates.
            offset (int): An optional pixel margin to add/subtract from the box.

        Returns:
            An np.ndarray containing the cropped object image.
        """
        x1, y1, x2, y2 = map(int, box_xyxy)
        x1, y1 = max(x1 - offset, 0), max(y1 - offset, 0)
        x2, y2 = min(x2 + offset, image.shape[1]), min(y2 + offset, image.shape[0])
        return image[y1:y2, x1:x2]

# ==================================
#      YoloDetection (Subclass)
# ==================================
class YoloDetection(YoloObjectBase):
    """
    A specialized class for standard object detection.
    Inherits all prediction and tracking logic from YoloObjectBase.
    """
    def __init__(self, model_path: str, warmup: bool = True) -> None:
        """
        Initializes the detection model.

        Args:
            model_path (str): Path to the detection model file.
            warmup (bool): If True, runs a dummy inference to warm up the model.
        """
        super().__init__(model_path, task='detect', warmup=warmup)

    def decode_results(self, results: Union[Results, List[Results]]) -> Union[list, List[list]]:
        """
        Decodes raw ultralytics Results into a simplified list format.

        Args:
            results: A single Results object or a list of them.

        Returns:
            For a single result: A list of [(box_coords, score, class_name)].
            For a list of results: A list of lists, one for each result.
        """
        if isinstance(results, list):
            return [self._decode_single_result(res) for res in results]
        else:
            return self._decode_single_result(results)

    def _decode_single_result(self, result: Results) -> list:
        """Helper for decoding one ultralytics.Results object."""
        boxes = result.boxes
        if not boxes: return []
        box_coords = [[int(v) for v in box] for box in boxes.xyxy.tolist()]
        scores = boxes.conf.tolist()
        class_names = [self.class_names[int(id)] for id in boxes.cls.tolist()]
        return list(zip(box_coords, scores, class_names))

    def decode_detections(self, detections: Detections) -> list:
        """
        Decodes a supervision.Detections object (from tracking) into a simplified list.

        Args:
            detections (Detections): A supervision.Detections object, typically with tracker IDs.

        Returns:
            A list of [(tracker_id, box_coords, score, class_name)].
        
        Raises:
            ValueError: If the Detections object does not contain tracker_id.
        """
        if detections.is_empty(): return []
        if detections.tracker_id is None: raise ValueError("Input Detections object has no tracker_id.")
        
        tracker_ids = detections.tracker_id.tolist()
        box_coords = [[int(v) for v in box] for box in detections.xyxy.tolist()]
        scores = detections.confidence.tolist()
        class_names = [self.class_names[id] for id in detections.class_id.tolist()]
        return list(zip(tracker_ids, box_coords, scores, class_names))

# ==================================
#     YoloSegmentation (Subclass)
# ==================================
class YoloSegmentation(YoloObjectBase):
    """
    A specialized class for instance segmentation.
    Inherits all prediction and tracking logic from YoloObjectBase.
    """
    def __init__(self, model_path: str, warmup: bool = True) -> None:
        """
        Initializes the segmentation model.

        Args:
            model_path (str): Path to the segmentation model file.
            warmup (bool): If True, runs a dummy inference to warm up the model.
        """
        super().__init__(model_path, task='segment', warmup=warmup)

    def decode_results(self, results: Union[Results, List[Results]]) -> Union[list, List[list]]:
        """
        Decodes raw ultralytics Results into a simplified list format.

        Args:
            results: A single Results object or a list of them.

        Returns:
            For a single result: A list of [(box_coords, polygon_mask, score, class_name)].
            For a list of results: A list of lists, one for each result.
        """
        if isinstance(results, list):
            return [self._decode_single_result(res) for res in results]
        else:
            return self._decode_single_result(results)

    def _decode_single_result(self, result: Results) -> list:
        """Helper for decoding one ultralytics.Results object with segmentation masks."""
        if result.masks is None: return []
        boxes = result.boxes
        masks = [segment.astype(int) for segment in result.masks.xy]
        box_coords = [[int(v) for v in box] for box in boxes.xyxy.tolist()]
        scores = boxes.conf.tolist()
        class_names = [self.class_names[int(id)] for id in boxes.cls.tolist()]
        return list(zip(box_coords, masks, scores, class_names))

    def decode_detections(self, detections: Detections) -> list:
        """
        Decodes a supervision.Detections object (from tracking) into a simplified list.

        Args:
            detections (Detections): A supervision.Detections object, typically with tracker IDs and masks.

        Returns:
            A list of [(tracker_id, box_coords, binary_mask, score, class_name)].
        
        Raises:
            ValueError: If the Detections object does not contain tracker_id or mask data.
        """
        if detections.is_empty(): return []
        if detections.tracker_id is None: raise ValueError("Input Detections object has no tracker_id.")
        if detections.mask is None: raise ValueError("Input Detections object has no segmentation masks.")

        tracker_ids = detections.tracker_id.tolist()
        box_coords = [[int(v) for v in box] for box in detections.xyxy.tolist()]
        masks = [m for m in detections.mask]
        scores = detections.confidence.tolist()
        class_names = [self.class_names[id] for id in detections.class_id.tolist()]
        return list(zip(tracker_ids, box_coords, masks, scores, class_names))

    def segment_object(self, image: np.ndarray, box_xyxy: tuple, mask: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        Extracts a segmented object from an image using its mask and bounding box.
        This method intelligently handles both polygon and binary mask formats.

        Args:
            image (np.ndarray): The source image.
            box_xyxy (tuple): The bounding box (x1, y1, x2, y2) of the object.
            mask (np.ndarray): The object's mask, either as polygon points or a binary mask.
            offset (int): An optional pixel margin to add to the final crop.

        Returns:
            An np.ndarray containing the cropped, segmented object.
        """
        if mask.dtype == bool:
            binary_mask = mask.astype(np.uint8) * 255
        else:
            binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(binary_mask, [mask.astype(int)], -1, (255), thickness=cv2.FILLED)

        isolated_object = cv2.bitwise_and(image, image, mask=binary_mask)
        return self.extract_object(isolated_object, box_xyxy, offset)

# ==================================
#        YoloPose (Subclass)
# ==================================
class YoloPose(YoloObjectBase):
    """
    A specialized class for pose estimation.
    Inherits all prediction and tracking logic from YoloObjectBase.
    """
    def __init__(self, model_path: str, warmup: bool = True) -> None:
        """
        Initializes the pose estimation model.

        Args:
            model_path (str): Path to the pose estimation model file.
            warmup (bool): If True, runs a dummy inference to warm up the model.
        """
        super().__init__(model_path, task='pose', warmup=warmup)

    def decode_results(self, results: Union[Results, List[Results]]) -> Union[list, List[list]]:
        """
        Decodes raw ultralytics Results into a simplified list format.

        Returns:
            A list of [(box_coords, keypoints, score, class_name)] for each instance.
            The 'keypoints' are a numpy array of shape (num_keypoints, 2) with [x, y] coordinates.
        """
        if isinstance(results, list):
            return [self._decode_single_result(res) for res in results]
        else:
            return self._decode_single_result(results)

    def _decode_single_result(self, result: Results) -> list:
        """Helper for decoding one ultralytics.Results object."""
        # Ensure there are keypoints to process
        if result.keypoints is None: 
            return []
        
        boxes = result.boxes
        
        # Extract each piece of data into its own variable
        box_coords = [[int(v) for v in box] for box in boxes.xyxy.tolist()]
        scores = boxes.conf.tolist()
        class_names = [self.class_names[int(id)] for id in boxes.cls.tolist()]
        
        # Extract keypoints (num_instances, num_keypoints, 2)
        keypoints_data = result.keypoints.xy.cpu().numpy().astype(int)
        keypoints_list = [kps for kps in keypoints_data] # Convert to a list of arrays

        return list(zip(box_coords, keypoints_list, scores, class_names))

    def decode_detections(self, detections: Detections) -> list:
        """
        Decodes a supervision.Detections object (from tracking) into a simplified list.

        Returns:
            A list of [(tracker_id, box_coords, keypoints, score, class_name)].
            The 'keypoints' are a numpy array of shape (num_keypoints, 2) with [x, y] coordinates.
        """
        # Ensure there are detections to process
        if detections.is_empty(): 
            return []
        if detections.tracker_id is None: 
            raise ValueError("Input Detections object does not have tracker_id.")
        # `supervision` stores keypoints in the 'data' dictionary under 'keypoints'
        if 'keypoints' not in detections.data:
            raise ValueError("Input Detections object does not have keypoints data.")

        # Extract each piece of data into its own variable
        tracker_ids = detections.tracker_id.tolist()
        box_coords = [[int(val) for val in box] for box in detections.xyxy.tolist()]
        scores = detections.confidence.tolist()
        class_names = [self.class_names[id] for id in detections.class_id.tolist()]
        
        # Extract keypoints from the 'data' attribute
        keypoints_data = detections.data['keypoints'].astype(int)
        keypoints_list = [kps for kps in keypoints_data] # Convert to a list of arrays
        
        return list(zip(tracker_ids, box_coords, keypoints_list, scores, class_names))
    