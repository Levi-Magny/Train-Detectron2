"""
Exemplo de uso da biblioteca finetune-detectron com o dataset Balloon.

Este script converte as anotações VIA do Balloon para COCO, treina o modelo,
e realiza uma predição em uma imagem de exemplo.
"""

import os
from finetune_detectron import DetectronTrainer

def main():
    # Caminhos do dataset Balloon
    trainer = DetectronTrainer("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    balloon_dir = "balloon"
    train_dir = os.path.join(balloon_dir, "train")
    val_dir = os.path.join(balloon_dir, "val")
    via_train_json = os.path.join(train_dir, "via_region_data.json")
    via_val_json = os.path.join(val_dir, "via_region_data.json")

    # Caminhos para JSON COCO convertidos
    coco_train_json = os.path.join(train_dir, "coco_annotations.json")
    coco_val_json = os.path.join(val_dir, "coco_annotations.json")

    # Converter VIA para COCO
    trainer.convert_via_to_coco(via_json_path=via_train_json, images_dir=train_dir, output_coco_json=coco_train_json)
    trainer.convert_via_to_coco(via_json_path=via_val_json, images_dir=val_dir, output_coco_json=coco_val_json)

    # Treinar o modelo
    trainer.train(
        dataset_name="balloon",
        json_train=coco_train_json,
        images_train=train_dir,
        json_val=coco_val_json,
        images_val=val_dir,
        output_dir="./output",
        num_classes=1,
        max_iter=300,
        device="cpu"
    )

    # Exemplo de predição
    # Assumindo que há uma imagem de exemplo
    image_path = "test.jpg"  # Substitua por uma imagem real
    if os.path.exists(image_path):
        results = trainer.predict(
            image_path=image_path,
            model_weights="./output/model_final.pth",
            num_classes=1,
            device="cpu"
        )

        # Salvar a imagem com predições
        trainer.save_prediction_image(results, "prediction_result.jpg")
        print("Predição salva em prediction_result.jpg")
    else:
        print(f"Imagem {image_path} não encontrada. Adicione uma imagem para testar a predição.")

if __name__ == "__main__":
    main()