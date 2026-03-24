# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import cv2
import os
import json
import random
import numpy as np
from typing import Any

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.utils.events import TensorboardXWriter
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer

class MyTrainer(DefaultTrainer):
    def build_writers(self):
        # Adiciona TensorboardXWriter aos writers padrão
        writers = super().build_writers()
        writers.append(TensorboardXWriter(self.cfg.OUTPUT_DIR))
        return writers

def get_balloon_dicts(img_dir: str)->list:
    """Load a subset of the balloon dataset in detectron2 format.
    
    Args:
        img_dir (str): Path to the directory containing the images and annotations.

    Returns:
        list: List of dataset dictionaries.
    """
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2] # type: ignore

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def visualize_sample(dataset_dicts: list, metadata: Any)->None:
    """Visualize a random sample from the dataset.
    
    Args:
        dataset_dicts (list): List of dataset dictionaries.
        metadata (MetadataCatalog): Metadata for the dataset."""
    sample = random.choice(dataset_dicts)
    img = cv2.imread(sample["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5) # type: ignore
    vis = visualizer.draw_dataset_dict(sample)
    cv2.imshow("Sample", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)

def finetune_model():
    """a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset. 300 iterations on a CPU."""
    
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"  # Forçar execução na CPU
    
    cfg.OUTPUT_DIR = "./output"  # Diretório para salvar logs e checkpoints
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    dataset_dicts = get_balloon_dicts("balloon/train")
    finetune_model()

