import supervision as sv
import numpy as np
from typing import Union, List, Any, Tuple
from ultralytics import YOLO
from supervision.detection.core import Detections
from ultralytics.engine.results import Results

class YoloModel:
    """
    A base class for wrapping Ultralytics YOLO models.
    """
    def __init__(self, model:str, task:str) -> None:
        self.model = YOLO(model, task=task)

    def decode_speed(self, results: Results) -> dict:
        return results.speed

    @property
    def class_names(self) -> dict:
        """Returns the dictionary mapping class indices to class names."""
        return self.model.names
    
class YoloClassification(YoloModel):
    """
    A specialized class for YOLO classification tasks.
    """
    def __init__(self, model_path: str, warmup: bool = True) -> None:
        super().__init__(model_path, task='classify')
        if warmup:
            # Warm up the model with a realistic input size.
            # Most classification models use 224x224.
            print("Warming up the classification model...")
            dummy_input = np.zeros((1, 1, 3), dtype=np.uint8)
            self.model.predict(dummy_input, verbose=False)
            print("Warm-up complete.")
    
    def _process_top1(self, result: Results) -> Tuple[str, float]:
        """Helper to extract top-1 result from a single prediction."""
        idx = result.probs.top1
        return result.names[idx], result.probs.top1conf.item()

    def _process_top5(self, result: Results) -> List[Tuple[str, float]]:
        """Helper to extract top-5 results from a single prediction."""
        indices = result.probs.top5
        names = [result.names[idx] for idx in indices]
        scores = [result.probs.data[idx].item() for idx in indices]
        return list(zip(names, scores))

    def classify_top1(self, source: Union[np.ndarray, List[np.ndarray]], **kwargs: Any) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
        """
        Performs classification and returns the top-1 class and score.

        Returns:
            - For a single image: A single (class_name, score) tuple.
            - For a list of images: A list of (class_name, score) tuples.
        """
        is_batch = isinstance(source, list)
        results = self.model.predict(source, verbose=False, **kwargs)
        
        processed_results = [self._process_top1(res) for res in results]

        return processed_results if is_batch else processed_results[0]
    
    def classify_top5(self, source: Union[np.ndarray, List[np.ndarray]], **kwargs: Any) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """
        Performs classification and returns the top-5 classes and scores.

        Returns:
            - For a single image: A list of the top 5 (class_name, score) tuples.
            - For a list of images: A list containing a list of top 5 tuples for each image.
        """
        is_batch = isinstance(source, list)
        results = self.model.predict(source, verbose=False, **kwargs)

        processed_results = [self._process_top5(res) for res in results]

        return processed_results if is_batch else processed_results[0]

class YoloObjectDetector(YoloModel):
    def __init__(self, model):
        super().__init__(model, task='detect')
        self.predict(np.zeros((1, 1, 3), dtype=np.uint8))
        self.tracker = sv.ByteTrack()
    
    # --- Methods ---
    def get_speed(self, results:Results) -> dict:
        return results.speed

    def get_image_size(self, results:Results) -> Tuple[int, int]:
        return results.orig_shape
    
    def get_info_supervision(self, results:Detections) -> list:
        obj_id = results.tracker_id.tolist()
        box = [[int(val) for val in box] for box in results.xyxy.tolist()] 
        score = results.confidence.tolist()
        cls_name = [self.cls_dict[id] for id in results.class_id.tolist()] 
        return list(zip(obj_id, box, score, cls_name))
    
    def get_info(self, results:Results, mode='xyxy') -> list:
        if not mode in ['xyxy', 'xyxyn', 'xywh', 'xywhn']:
            raise ValueError(f"mode must be ['xyxy', 'xyxyn', 'xywh', 'xywhn']")
        
        boxes = results.boxes

        box = [[int(val) for val in box] for box in getattr(boxes, mode).tolist()] if mode in ['xyxy', 'xywh'] else getattr(boxes, mode).tolist()
        score = boxes.conf.tolist()
        cls_name = [self.cls_dict[id] for id in boxes.cls.tolist()]

        if boxes.is_track:
            obj_id = map(int, boxes.id.tolist())
            return list(zip(obj_id, box, score, cls_name))

        else:
            return list(zip(box, score, cls_name))
        
    def tracking_supervision(self, source, conf:float=0.5, iou:float=0.5, max_det=100, agnostic_nms=True, **kwargs:Any) -> list:
        
        results:Results = self.model(source, conf=conf, iou=iou, max_det=max_det, agnostic_nms=agnostic_nms, verbose=False, **kwargs)[0]
        speed = self.get_speed(results)
        bboxes = results.boxes.xyxy.cpu().numpy() # Bounding boxes (x1, y1, x2, y2)
        scores = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy() 

        detections_list = sv.Detections(
            xyxy=bboxes,
            confidence=scores,
            class_id=cls_ids,
        )

        # Update tracker with detections
        tracked_objects = self.tracker.update_with_detections(detections_list)
        return speed, tracked_objects
    
        track_id = tracked_objects.tracker_id.tolist()
        track_xyxy = tracked_objects.xyxy.tolist()
        track_score = tracked_objects.confidence.tolist()
        track_cls_id = tracked_objects.class_id.tolist()

        detection_results = []
        for obj_xyxy, obj_id, obj_score, obj_cls_id in zip(track_xyxy, track_id, track_score, track_cls_id):
            detection_results.append(obj_xyxy + [obj_id] + [obj_score] + [obj_cls_id])

        return detection_results

    def tracking(self, source, conf:float=0.5, iou:float=0.5, max_det=100, agnostic_nms=True, **kwargs:Any) -> Results:
        """
        Ultralytic defaults
        max_det=300
        agnostic_nms=False
        """
        return self.model.track(source, conf=conf, iou=iou, max_det=max_det, agnostic_nms=agnostic_nms, verbose=False, **kwargs)[0]
    
    def predict(self, source, conf:float=0.5, iou:float=0.5, max_det=100, agnostic_nms=True, **kwargs:Any) -> Results:
        """
        Ultralytic defaults
        max_det=300
        agnostic_nms=False
        """
        return self.model(source, conf=conf, iou=iou, max_det=max_det, agnostic_nms=agnostic_nms, verbose=False, **kwargs)[0]
    
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
