# FineTune-Detectron2

Biblioteca simples para finetune e inferência com Detectron2 (Mask R-CNN) em datasets COCO.

## Organização do projeto

- `/src`: código principal (`finetune_detectron.py`, `example_usage.py`, `loading_dataset_example.py`).
- `/tests`: testes automatizados `pytest`.
- `/doc`: documentação de exemplo detalhada.

## Dependências principais

- Python 3.8+
- Detectron2 (v0.6+cpu no exemplo)
- PyTorch 1.10.1+cpu
- torchvision 0.11.2+cpu
- OpenCV `opencv-python`
- NumPy
- pycocotools
- pytest

Versões usadas aqui (ambiente documentado):

- detectron2==0.6+cpu
- torch==1.10.1+cpu
- torchvision==0.11.2+cpu
- numpy==1.24.4
- opencv-python==4.8.0.74
- pytest==7.x

> Ajuste de acordo com o `pip freeze` do seu ambiente.

## Instalação

1. Clone o repositório e entre no diretório:

```bash
git clone <url-do-seu-repo>
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

4. Instale o `detectron2` compilado conforme sua plataforma (se não via `requirements`):

```bash
pip install detectron2==0.6+cpu -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch1.13/index.html
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

