import json
import os
from pathlib import Path
import sys
import pytest

import cv2
import numpy as np

# Adicionar src no path para importação do pacote local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from finetune_detectron import DetectronTrainer


def test_convert_via_to_coco_generates_coco_json(tmp_path: Path):
    via_json = tmp_path / "via.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Criar imagem de teste (100x100 preta)
    image_path = images_dir / "img1.jpg"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    # Criar anotação VIA para um quadrado simples
    via_data = {
        "img1.jpg": {
            "filename": "img1.jpg",
            "size": 0,
            "regions": {
                "0": {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [20, 80, 80, 20],
                        "all_points_y": [20, 20, 80, 80]
                    },
                    "region_attributes": {}
                }
            },
            "file_attributes": {}
        }
    }

    via_json.write_text(json.dumps(via_data))
    output_json = tmp_path / "coco.json"

    DetectronTrainer.convert_via_to_coco(
        via_json_path=str(via_json),
        images_dir=str(images_dir),
        output_coco_json=str(output_json),
        categories=["balloon"]
    )

    assert output_json.exists()
    coco = json.loads(output_json.read_text())

    assert "images" in coco and len(coco["images"]) == 1
    assert "annotations" in coco and len(coco["annotations"]) == 1
    assert "categories" in coco and coco["categories"][0]["name"] == "balloon"

    ann = coco["annotations"][0]
    assert ann["category_id"] == 0
    assert ann["bbox"] == [20.0, 20.0, 60.0, 60.0]
    assert ann["area"] == 3600.0


def test_convert_via_to_coco_handles_no_annotations(tmp_path: Path):
    via_json = tmp_path / "via_empty.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image_path = images_dir / "img1.jpg"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    via_data = {
        "img1.jpg": {
            "filename": "img1.jpg",
            "size": 0,
            "regions": {},
            "file_attributes": {}
        }
    }

    via_json.write_text(json.dumps(via_data))
    output_json = tmp_path / "coco_empty.json"

    DetectronTrainer.convert_via_to_coco(
        via_json_path=str(via_json),
        images_dir=str(images_dir),
        output_coco_json=str(output_json),
        categories=["balloon"]
    )

    coco = json.loads(output_json.read_text())
    assert coco["images"]
    assert coco["annotations"] == []


def test_convert_yolo_to_coco_output_values(tmp_path: Path):
    images_dir = tmp_path / "images"
    yolo_dir = tmp_path / "yolo"
    images_dir.mkdir()
    yolo_dir.mkdir()

    # Criar imagem de teste (100x100) para a correspondência YOLO
    image_path = images_dir / "img1.jpg"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    # Criar anotação YOLO: classe 0, centro no meio, largura=0.6, altura=0.4
    yolo_annotation = "0 0.5 0.5 0.6 0.4\n"
    (yolo_dir / "img1.txt").write_text(yolo_annotation)

    output_json = tmp_path / "coco_yolo.json"

    DetectronTrainer.convert_yolo_to_coco(
        yolo_dir=str(yolo_dir),
        images_dir=str(images_dir),
        output_coco_json=str(output_json),
        categories=["balloon"]
    )

    assert output_json.exists()
    coco = json.loads(output_json.read_text())

    assert "images" in coco and len(coco["images"]) == 1
    assert "annotations" in coco and len(coco["annotations"]) == 1
    assert "categories" in coco and coco["categories"][0]["name"] == "balloon"

    ann = coco["annotations"][0]
    assert ann["category_id"] == 0

    expected_bbox = [20.0, 30.0, 60.0, 40.0]
    assert pytest.approx(ann["bbox"]) == expected_bbox
    assert pytest.approx(ann["area"]) == 2400.0


if __name__ == "__main__":
    import pytest

    pytest.main(["-q", __file__])

