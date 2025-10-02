# ==================================
#         Import Dependencies
# ==================================
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
#      YoloDetection (Subclass)
# ==================================
class YoloDetection(YoloModel):
    """
    A specialized class for YOLO object detection and tracking tasks
    using supervision for stateful tracking.
    """
    def __init__(self, model_path: str, warmup: bool = True) -> None:
        """
        Initializes the detection model and the stateful tracker.

        Args:
            model_path (str): Path to the detection model file.
            warmup (bool): If True, runs a dummy inference to warm up the model.
        """
        super().__init__(model_path, task='detect')
        if warmup:
            self.warmup(imgsz=self.training_imgsz or 640)
        
        # Initialize a stateful tracker for the instance
        self.tracker = sv.ByteTrack()

    def _set_default_kwargs(self, kwargs: Dict) -> Dict:
        """Helper to set default prediction arguments to avoid code repetition."""
        kwargs.setdefault('conf', 0.5)
        kwargs.setdefault('iou', 0.5)
        kwargs.setdefault('max_det', 50)
        kwargs.setdefault('agnostic_nms', True)
        return kwargs

    def _decode_single_result(self, result: Results, mode: str) -> list:
        """Helper for decoding one ultralytics.Results object."""
        if not mode in ['xyxy', 'xyxyn', 'xywh', 'xywhn']:
            raise ValueError(f"mode must be one of ['xyxy', 'xyxyn', 'xywh', 'xywhn']")
        
        boxes = result.boxes
        if not boxes: return []

        # Extract coordinates, scores, and class names
        box_coords = [[int(v) for v in box] for box in getattr(boxes, mode).tolist()] if mode in ['xyxy', 'xywh'] else getattr(boxes, mode).tolist()
        scores = boxes.conf.tolist()
        class_names = [self.class_names[int(id)] for id in boxes.cls.tolist()]
        
        return list(zip(box_coords, scores, class_names))

    def predict(self, source: Union[np.ndarray, List[np.ndarray]], **kwargs: Any) -> Union[Results, List[Results]]:
        """
        Runs stateless object detection. Handles both single and batch sources.

        Args:
            source: A single image (np.ndarray) or a list of images.
            **kwargs: Additional ultralytics predict arguments. If not provided,
                the following defaults are used:
                - conf (float): 0.5
                - iou (float): 0.5
                - max_det (int): 50
                - agnostic_nms (bool): True

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
                the following defaults are used:
                - conf (float): 0.5
                - iou (float): 0.5
                - max_det (int): 50
                - agnostic_nms (bool): True

        Returns:
            A supervision.Detections object with tracker IDs assigned.
        
        Raises:
            ValueError: If the source is a list (batch input).
        """
        if isinstance(source, list):
            raise ValueError("The 'tracking' method does not support batch inputs. Please process frames sequentially.")
        
        # Step 1: Set prediction arguments
        kwargs = self._set_default_kwargs(kwargs)

        # Step 2: Run prediction on the single frame
        results: Results = self.model.predict(source, verbose=False, **kwargs)[0]

        # Step 3: Convert the raw results to a supervision Detections object
        detections = Detections.from_ultralytics(results)

        # Step 4: Update the tracker with the new detections and get tracked objects
        tracked_detections = self.tracker.update_with_detections(detections)
        
        return tracked_detections
    
    def decode_results(self, results: Union[Results, List[Results]], mode: str = 'xyxy') -> Union[list, List[list]]:
        """
        Decodes raw ultralytics.Results into a simplified list format.

        Args:
            results: A single Results object or a list of them.
            mode (str): The bounding box format ('xyxy', 'xywh', 'xyxyn', 'xywhn').

        Returns:
            For a single result: A list of [(box_coords, score, class_name)].
            For a list of results: A list of lists, one for each result.
        """
        if isinstance(results, list):
            return [self._decode_single_result(res, mode) for res in results]
        else:
            return self._decode_single_result(results, mode)

    def decode_detections(self, detections: Detections) -> list:
        """
        Decodes a supervision.Detections object (output from tracking) into a simplified list.

        Args:
            detections (Detections): A supervision.Detections object, typically with tracker IDs.

        Returns:
            A list of [(tracker_id, box_coords, score, class_name)].
        
        Raises:
            ValueError: If the Detections object does not contain tracker_id.
        """
        if detections.is_empty(): 
            return []
            
        if detections.tracker_id is None:
            raise ValueError("Input Detections object does not have tracker_id.")

        # Extract tracker IDs, boxes, scores, and class names
        tracker_ids = detections.tracker_id.tolist()
        box_coords = [[int(val) for val in box] for box in detections.xyxy.tolist()] 
        scores = detections.confidence.tolist()
        class_names = [self.class_names[id] for id in detections.class_id.tolist()] 

        return list(zip(tracker_ids, box_coords, scores, class_names))

    def extract_object(self, image: np.ndarray, box_xyxy: Tuple[int, int, int, int], offset: int = 0) -> np.ndarray:
        """
        Extracts a detected object from an image using its bounding box.

        Args:
            image (np.ndarray): The source image.
            box_xyxy (tuple): A tuple of (x1, y1, x2, y2) coordinates.
            offset (int): An optional pixel offset to add/subtract from the box, creating a margin.

        Returns:
            An np.ndarray containing the cropped object image.
        """
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(int, box_xyxy)

        # Apply offset and clamp coordinates to be within the image boundaries
        x1 = max(x1 - offset, 0)
        y1 = max(y1 - offset, 0)
        x2 = min(x2 + offset, width)
        y2 = min(y2 + offset, height)

        # Crop and return the image
        return image[y1:y2, x1:x2]
    
class YoloSegmentation(YoloModel):
    def __init__(self, model):
        super().__init__(model, task='segment')
        self.predict(np.zeros((1, 1, 3), dtype=np.uint8))
        self.tracker = sv.ByteTrack()

    def predict(self, source, conf:float=0.5, iou:float=0.5, max_det=100, agnostic_nms=True, **kwargs:Any) -> Results:
        """
        Ultralytic defaults
        max_det=300
        agnostic_nms=False
        """
        return self.model(source, conf=conf, iou=iou, max_det=max_det, agnostic_nms=agnostic_nms, verbose=False, **kwargs)[0]
    
    def get_info(self, results:Results, mode='xyxy', cls_idx=False) -> list:
        if not mode in ['xyxy', 'xyxyn', 'xywh', 'xywhn']:
            raise ValueError(f"mode must be ['xyxy', 'xyxyn', 'xywh', 'xywhn']")
        
        boxes = results.boxes

        box = [[int(val) for val in box] for box in getattr(boxes, mode).tolist()] if mode in ['xyxy', 'xywh'] else getattr(boxes, mode).tolist()
        mask = [segment.astype(int) for segment in results.masks.xy]
        score = boxes.conf.tolist()
        if not cls_idx:
            cls_name = [self.cls_dict[id] for id in boxes.cls.tolist()]
        else:
            cls_name = [id for id in boxes.cls.tolist()]

        if boxes.is_track:
            obj_id = map(int, boxes.id.tolist())
            return list(zip(obj_id, box, mask, score, cls_name))

        else:
            return list(zip(box, mask, score, cls_name))
        
    def segment_detect(self, image:np.ndarray, box_xyxy:tuple, mask:np.ndarray, offset:int = 0) -> np.ndarray:
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(binary_mask, [mask], -1, (255), thickness=cv2.FILLED)
        isolated_object = cv2.bitwise_and(image, image, mask=binary_mask)
        crop_image = self.extract_detect(isolated_object, box_xyxy, offset)
        return crop_image
    
    # Extract image
    def extract_detect(self, image:np.ndarray, box_xyxy:tuple, offset:int = 0) -> np.ndarray:

        height, width = image.shape[:2]

        # Decode box coordinate tuple
        x1, y1, x2, y2 = box_xyxy

        # Processing coordinate with offset
        x1 = max(x1 - offset, 0) # prevent x1 become negative
        y1 = max(y1 - offset, 0) # prevent y1 become negative
        x2 = min(width, x2 + offset) # prevent x2 overflow from vdeo frame
        y2 = min(height, y2 + offset) # prevent y2 overflow from vdeo frame

        # Extract image
        extract_image = image[y1:y2, x1:x2]
        return extract_image
