"""
Módulo para treinamento e inferência com Detectron2.

Este módulo fornece funções para treinar um modelo Mask R-CNN R50-FPN pré-treinado no COCO
em um dataset personalizado no formato COCO, e para realizar predições em imagens.

Exemplo de uso:
    from finetune-detectron import train_model, predict_image

    # Treinar
    train_model(
        dataset_name="my_dataset",
        json_train="path/to/train.json",
        images_train="path/to/train/images",
        json_val="path/to/val.json",
        images_val="path/to/val/images",
        output_dir="./output",
        num_classes=1,
        max_iter=300
    )

    # Predizer
    results = predict_image(
        image_path="path/to/image.jpg",
        model_weights="./output/model_final.pth",
        num_classes=1
    )
"""

import cv2
import os
import json
import numpy as np
from typing import Optional, Dict, Any, List

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.events import TensorboardXWriter

# COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
# COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml

class DetectronTrainer:
    """
    Classe para treinamento e inferência com Detectron2 Mask R-CNN.

    Permite treinar um modelo pré-treinado no COCO e realizar predições em imagens.
    """

    def __init__(self, config_file: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        """
        Inicializa o treinador com um arquivo de configuração.

        Args:
            config_file (str): Caminho do arquivo de configuração no model_zoo.
        """
        self.config_file = config_file

    @staticmethod
    def convert_via_to_coco(via_json_path: str, images_dir: str, output_coco_json: str, categories: List[str] = ["balloon"]) -> None:
        """
        Converte anotações do VIA (VGG Image Annotator) para formato COCO.

        Args:
            via_json_path (str): Caminho para o JSON do VIA.
            images_dir (str): Diretório das imagens.
            output_coco_json (str): Caminho para salvar o JSON COCO.
            categories (list): Lista de nomes de categorias.
        """
        with open(via_json_path, 'r') as f:
            imgs_anns = json.load(f)

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(categories)]
        }

        annotation_id = 0
        for idx, v in enumerate(imgs_anns.values()):
            filename = os.path.join(images_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]  # type: ignore

            coco_data["images"].append({
                "id": idx,
                "file_name": v["filename"],
                "height": height,
                "width": width
            })

            annos = v["regions"]
            for _, anno in annos.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly_flat = [p for x in poly for p in x]

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": idx,
                    "category_id": 0,
                    "segmentation": [poly_flat],
                    "area": float((max(px) - min(px)) * (max(py) - min(py))),
                    "bbox": [float(min(px)), float(min(py)), float(max(px) - min(px)), float(max(py) - min(py))],
                    "iscrowd": 0
                })
                annotation_id += 1

        with open(output_coco_json, 'w') as f:
            json.dump(coco_data, f, indent=4)

    def train(
        self,
        dataset_name: str,
        json_train: str,
        images_train: str,
        json_val: Optional[str] = None,
        images_val: Optional[str] = None,
        output_dir: str = "./output",
        pretrained_weights: Optional[str] = None,
        num_classes: int = 1,
        max_iter: int = 300,
        batch_size: int = 2,
        learning_rate: float = 0.00025,
        device: str = "cpu"
    ) -> None:
        """
        Treina o modelo Mask R-CNN no dataset fornecido.

        Args:
            dataset_name (str): Nome do dataset (usado para registro).
            json_train (str): Caminho para o arquivo JSON de anotações de treino (formato COCO).
            images_train (str): Diretório das imagens de treino.
            json_val (str, optional): Caminho para o arquivo JSON de anotações de validação.
            images_val (str, optional): Diretório das imagens de validação.
            output_dir (str): Diretório para salvar checkpoints e logs.
            pretrained_weights (str, optional): Caminho para pesos pré-treinados. Se None, baixa do COCO.
            num_classes (int): Número de classes no dataset (excluindo background).
            max_iter (int): Número máximo de iterações de treinamento.
            batch_size (int): Tamanho do batch por GPU (IMS_PER_BATCH).
            learning_rate (float): Taxa de aprendizado base.
            device (str): Dispositivo para treinamento ("cpu" ou "cuda").
        """
        # Registrar datasets
        register_coco_instances(f"{dataset_name}_train", {}, json_train, images_train)
        if json_val and images_val:
            register_coco_instances(f"{dataset_name}_val", {}, json_val, images_val)

        # Configurar
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
        cfg.DATASETS.TEST = (f"{dataset_name}_val",) if json_val else ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = pretrained_weights or model_zoo.get_checkpoint_url(self.config_file)
        cfg.SOLVER.IMS_PER_BATCH = batch_size
        cfg.SOLVER.BASE_LR = learning_rate
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.DEVICE = device
        cfg.OUTPUT_DIR = output_dir

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def predict(
        self,
        image_path: str,
        model_weights: str,
        num_classes: int = 1,
        score_thresh: float = 0.7,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Realiza predição em uma imagem usando o modelo treinado.

        Args:
            image_path (str): Caminho para a imagem de entrada.
            model_weights (str): Caminho para os pesos do modelo treinado (.pth).
            num_classes (int): Número de classes no modelo.
            score_thresh (float): Limiar de pontuação para predições.
            device (str): Dispositivo para inferência ("cpu" ou "cuda").

        Returns:
            dict: Resultados da predição, incluindo caixas, máscaras, etc.
        """
        # Configurar
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.WEIGHTS = model_weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.DEVICE = device

        # Carregar imagem
        im = cv2.imread(image_path)
        if im is None:
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        # Preditor
        predictor = DefaultPredictor(cfg)
        outputs = predictor(im)

        print(f"Número de instâncias detectadas: {len(outputs['instances'])}")

        return {
            "image": im,
            "outputs": outputs,
            "visualized": self._visualize_predictions(im, outputs)
        }

    def _visualize_predictions(self, image: Any, outputs: Any) -> Any:
        """
        Visualiza as predições na imagem.

        Args:
            image: Imagem OpenCV.
            outputs: Saídas do modelo.

        Returns:
            Imagem visualizada.
        """
        v = Visualizer(image[:, :, ::-1], scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]

    def save_prediction_image(self, prediction_result: Dict[str, Any], output_path: str) -> None:
        """
        Salva a imagem com predições visualizadas e um arquivo de texto com bounding boxes e segmentações.

        Args:
            prediction_result (dict): Resultado de predict().
            output_path (str): Caminho para salvar a imagem (adiciona .jpg se necessário).
        """
        # Salvar imagem visualizada
        image_path = output_path if output_path.endswith(('.jpg', '.png')) else f"{output_path}.jpg"
        cv2.imwrite(image_path, prediction_result["visualized"])

        # Salvar dados de bounding boxes e segmentações em texto
        text_path = output_path.rsplit('.', 1)[0] + '_predictions.txt'
        instances = prediction_result["outputs"]["instances"].to("cpu")
        with open(text_path, 'w') as f:
            f.write("Predições:\n")
            for i, (box, mask) in enumerate(zip(instances.pred_boxes, instances.pred_masks)):
                f.write(f"Objeto {i+1}:\n")
                f.write(f"  Bounding Box: {box.tolist()}\n")
                f.write(f"  Segmentação: {mask.tolist()}\n")
                f.write("\n")

    @staticmethod
    def convert_yolo_to_coco(yolo_dir: str, images_dir: str, output_coco_json: str, categories: List[str] = ["balloon"]) -> None:
        """
        Converte anotações do formato YOLO para formato COCO.

        Args:
            yolo_dir (str): Diretório contendo arquivos de anotação YOLO (.txt).
            images_dir (str): Diretório das imagens.
            output_coco_json (str): Caminho para salvar o JSON COCO.
            categories (list): Lista de nomes de categorias.
        """
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(categories)]
        }

        annotation_id = 0
        for idx, filename in enumerate(os.listdir(images_dir)):
            if not filename.lower().endswith(('.jpg', '.png')):
                continue

            image_path = os.path.join(images_dir, filename)
            height, width = cv2.imread(image_path).shape[:2]  # type: ignore

            coco_data["images"].append({
                "id": idx,
                "file_name": filename,
                "height": height,
                "width": width
            })

            yolo_annotation_path = os.path.join(yolo_dir, f"{os.path.splitext(filename)[0]}.txt")
            if os.path.exists(yolo_annotation_path):
                with open(yolo_annotation_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id, x_center, y_center, w, h = map(float, parts)
                        category_id = int(class_id)

                        x_min = (x_center - w / 2) * width
                        y_min = (y_center - h / 2) * height

                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": idx,
                            "category_id": category_id,
                            "area": float(w * h * width * height),
                            "bbox": [float(x_min), float(y_min), float(w * width), float(h * height)],
                            "iscrowd": 0
                        })
                        annotation_id += 1
        with open(output_coco_json, 'w') as f:
            json.dump(coco_data, f, indent=4)

    def evaluate_model(
        self,
        dataset_name: str,
        json_val: str,
        images_val: str,
        model_weights: str,
        num_classes: int = 1,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Avalia o modelo no dataset de validação.

        Args:
            dataset_name (str): Nome do dataset.
            json_val (str): JSON de validação.
            images_val (str): Diretório de imagens de validação.
            model_weights (str): Pesos do modelo.
            num_classes (int): Número de classes.
            device (str): Dispositivo.

        Returns:
            dict: Métricas de avaliação.
        """
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        # Registrar dataset
        register_coco_instances(f"{dataset_name}_val", {}, json_val, images_val)

        # Configurar
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        cfg.DATASETS.TEST = (f"{dataset_name}_val",)
        cfg.MODEL.WEIGHTS = model_weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.DEVICE = device

        # Avaliar
        evaluator = COCOEvaluator(f"{dataset_name}_val", cfg, False, output_dir="./output")
        val_loader = build_detection_test_loader(cfg, f"{dataset_name}_val")
        results = inference_on_dataset(DefaultPredictor(cfg), val_loader, evaluator)

        return results


# Funções de conveniência para uso direto
def train_model(**kwargs) -> None:
    """
    Função de conveniência para treinar o modelo.

    Args: Veja DetectronTrainer.train().
    """
    trainer = DetectronTrainer()
    trainer.train(**kwargs)


def predict_image(**kwargs) -> Dict[str, Any]:
    """
    Função de conveniência para predição.

    Args: Veja DetectronTrainer.predict().

    Returns: Resultado da predição.
    """
    trainer = DetectronTrainer()
    return trainer.predict(**kwargs)


def evaluate_model(**kwargs) -> Dict[str, Any]:
    """
    Função de conveniência para avaliação.

    Args: Veja DetectronTrainer.evaluate_model().

    Returns: Métricas de avaliação.
    """
    trainer = DetectronTrainer()
    return trainer.evaluate_model(**kwargs)


def convert_via_to_coco(**kwargs) -> None:
    """
    Função de conveniência para conversão VIA para COCO.

    Args: Veja DetectronTrainer.convert_via_to_coco().
    """
    DetectronTrainer.convert_via_to_coco(**kwargs)
