# FineTune-Detectron2

If you ever tried to train or finetune models with Detectron2 you may have had problems while setting up the local environment. The library is very version sensitive, which makes everything much more difficult.

This is a library made to finetune models with Detectron2 (Mask R-CNN) in COCO datasets. It includes conversion of other dataset standards (such as YOLO and VIA) to COCO format.

## Project 

- `/src`: Main code (`finetune_detectron.py`).
- `/examples`: Usage examples (`example_use.py`, `loading_dataset_example.py`).
- `/tests`: Automated tests `pytest`.
- `/doc`: docs and detailed example.

## Main Dependencies

- Python 3.8+
- Detectron2
- PyTorch
- torchvision
- OpenCV `opencv-python`
- NumPy
- pycocotools
- pytest


## Installation

### Option 1: Manual installation

1. Clone the repo and enter the folder:

```bash
git clone https://github.com/Levi-Magny/Train-Detectron2
cd <local-repo>/detectron2
```

2. Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Install `Pytorch` from the official website:

```bash
# CPU
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

5. Install a compiled version of `detectron2` according to your platform:

```bash
# CPU
pip install detectron2==0.6 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# CUDA 11.1 - torch 1.10
python -m pip install detectron2==0.6 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

Or compile it manually from the official repository.

### Option 2: Install as package (recommended)

Install directly from git (includes all dependencies except detectron2):

```bash
pip install git+https://github.com/Levi-Magny/Train-Detectron2.git
```

Then install detectron2 separately as above.

For development:

```bash
pip install "git+https://github.com/Levi-Magny/Train-Detectron2.git#egg=finetune-detectron2[dev]"
```

## Execution

- Tests:

```bash
pytest -q
```

- Quick example:

```bash
python examples/example_use.py
```

## Structure and functionalities

- `src/finetune_detectron.py`:
  - `DetectronTrainer.convert_via_to_coco`
  - `DetectronTrainer.train`
  - `DetectronTrainer.predict`
  - `DetectronTrainer.save_prediction_image`
  - `DetectronTrainer.evaluate_model`
  - funis de conveniência: `train_model`, `predict_image`, `evaluate_model`, `convert_via_to_coco`

- `src/example_usage.py`:
  - Conversion of Balloon VIA → COCO
  - Training
  - inference and result storing

## How to contribute

Fork the project and create a new branch, add tests for each new feature and open a PR. Maintain consistency with PEP8 and messages in english.

