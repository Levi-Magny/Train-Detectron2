# Exemplo de Uso - FineTune-Detectron2

Este documento mostra um passo a passo para usar as funcionalidades básicas do projeto: converter VIA para COCO, treinar e fazer inferência.

## 1) Preparar ambiente

1. Crie e ative o venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instale dependências:

```bash
pip install -r requirements.txt
```

3. Instale `detectron2` de acordo com sua plataforma:

- Para CPU:

```bash
pip install detectron2==0.6+cpu -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch1.13/index.html
```

- Para CUDA (exemplo CUDA 11.8 + PyTorch 1.13):

```bash
pip install detectron2==0.6+cu118 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch1.13/index.html
```

- Detecção CUDA disponível:

```python
import torch
print(torch.cuda.is_available())
```

Se retornar `True`, use o pacote CUDA adequado; se `False`, use `+cpu`.

4. Releases pré-compiladas e compatibilidade:

- Acesse https://github.com/facebookresearch/detectron2/releases
- Escolha a versão que combine com sua versão do PyTorch e CUDA:
  - `detectron2==0.6+cpu` (sem CUDA)
  - `detectron2==0.6+cu118` (CUDA 11.8)
  - `detectron2==0.6+cu117` (CUDA 11.7)
- As versões pré-compiladas evitam problemas de compilação com `gcc/g++` em sistemas variados.

## 2) Converter VIA para COCO

Supondo que você tenha o dataset `balloon/train/via_region_data.json` e imagens em `balloon/train/`:

```bash
python src/example_usage.py --convert
```

(ou use a classe diretamente em Python):

```python
from finetune_detectron import DetectronTrainer

DetectronTrainer.convert_via_to_coco(
    via_json_path='balloon/train/via_region_data.json',
    images_dir='balloon/train',
    output_coco_json='balloon/train/coco_annotations.json',
    categories=['balloon']
)
```

## 3) Treinar modelo

No script de exemplo:

```bash
python src/example_usage.py --train
```

Em código:

```python
trainer = DetectronTrainer(
    
    # Ajuste conforme necessidade
    model_config='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    num_classes=1,
    output_dir='output'
)

trainer.train(
    train_json='balloon/train/coco_annotations.json',
    val_json='balloon/val/coco_annotations.json',
    images_dir='balloon',
    max_iter=1000,
    lr=0.00025
)
```

## 4) Inferência e salvar imagem com bbox

```python
predictions = trainer.predict_image('input.jpg')
trainer.save_prediction_image(
    'input.jpg',
    predictions,
    output_path='output/predicted.jpg'
)
```

## 5) Avaliação

```python
trainer.evaluate_model(
    model_path='output/model_final.pth',
    val_json='balloon/val/coco_annotations.json',
    val_images_dir='balloon/val'
)
```

## 6) Rodar testes

```bash
pytest -q
```

---

Dica: mantenha a mesma estrutura de diretórios `balloon/train`, `balloon/val` e o JSON COCO gerar para evitar mismatch de paths.