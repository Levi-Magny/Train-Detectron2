"""
Example of using the finetune-detectron library with the Balloon dataset.

This script converts the VIA annotations of the Balloon dataset to COCO format, trains the model,
and performs a prediction on an example image.
"""

import os
from finetune_detectron2.finetune_detectron import DetectronTrainer

def main():
    # Paths to the Balloon dataset
    trainer = DetectronTrainer("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    balloon_dir = "balloon"
    train_dir = os.path.join(balloon_dir, "train")
    val_dir = os.path.join(balloon_dir, "val")
    via_train_json = os.path.join(train_dir, "via_region_data.json")
    via_val_json = os.path.join(val_dir, "via_region_data.json")

    # Path to the converted COCO JSON files
    coco_train_json = os.path.join(train_dir, "coco_annotations.json")
    coco_val_json = os.path.join(val_dir, "coco_annotations.json")

    # Convert VIA to COCO
    trainer.convert_via_to_coco(via_json_path=via_train_json, images_dir=train_dir, output_coco_json=coco_train_json)
    trainer.convert_via_to_coco(via_json_path=via_val_json, images_dir=val_dir, output_coco_json=coco_val_json)

    # Train the model
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

    # Prediction example
    # Assuning there is an example image
    image_path = "test.jpg"  # Replace with a real image path for testing
    if os.path.exists(image_path):
        results = trainer.predict(
            image_path=image_path,
            model_weights="./output/model_final.pth",
            num_classes=1,
            device="cpu"
        )

        # Save the image with predictions
        trainer.save_prediction_image(results, "prediction_result.jpg")
        print("Prediction saved in prediction_result.jpg")
    else:
        print(f"Image {image_path} not found. Add an image in order to test the prediction.")

if __name__ == "__main__":
    main()