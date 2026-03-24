# FineTune-Detectron2

Biblioteca simples para finetune e inferência com Detectron2 (Mask R-CNN) em datasets COCO.

## Organização do projeto

- `/src`: código principal (`finetune_detectron.py`, `example_usage.py`, `loading_dataset_example.py`).
- `/tests`: testes automatizados `pytest`.
- `/doc`: documentação de exemplo detalhada.

## Dependências principais

- Python 3.8+
- Detectron2
- PyTorch 1.10.1+cpu
- torchvision 0.11.2+cpu
- OpenCV `opencv-python`
- NumPy
- pycocotools
- pytest

Para o treinamento oficial devem ser usadas as versões com cuda, veja em detalhes abaixo.

> Ajuste de acordo com o `pip freeze` do seu ambiente.

## Instalação

1. Clone o repositório e entre no diretório:

```bash
git clone https://github.com/Levi-Magny/Train-Detectron2
cd <repo>/detectron2
```

2. Crie e ative ambiente virtual (recomendado):

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Instale dependências:

```bash
pip install -r requirements.txt
```

4. Instale o `Pytorch` a partir do site oficial:

```bash
# CPU
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

5. Instale o `detectron2` compilado conforme sua plataforma:

```bash
# CPU
pip install detectron2==0.6 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# CUDA 11.1 - torch 1.10
python -m pip install detectron2==0.6 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

ou usando a release que você compilou manualmente (caminho local ou URL específico).

## Execução

- Testes:

```bash
pytest -q
```

- Uso rápido (script de exemplo):

```bash
python src/example_usage.py
```

## Estrutura de funcionalidades

- `src/finetune_detectron.py`:
  - `DetectronTrainer.convert_via_to_coco`
  - `DetectronTrainer.train`
  - `DetectronTrainer.predict`
  - `DetectronTrainer.save_prediction_image`
  - `DetectronTrainer.evaluate_model`
  - funis de conveniência: `train_model`, `predict_image`, `evaluate_model`, `convert_via_to_coco`

- `src/example_usage.py`:
  - fluxo de conversão Balloon VIA → COCO
  - treinamento
  - inferência e gravação de resultados

## Como contribuir

Crie um novo branch, adicione testes para cada novo comportamento e abra PR. Mantenha consistência de linha de estilo PEP8 e mensagens em português ou inglês conforme o padrão do projeto.

