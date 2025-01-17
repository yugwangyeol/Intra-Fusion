# Intra-Fusion

<div align="center">
  <img src="https://github.com/user-attachments/assets/a7ebb6ce-8aac-4692-b7ce-65fcbebe8c0e" alt="Intra-Fusion">
</div>

## Project Overview
이 프로젝트는 ICLR 2024에서 발표된 "Towards Meta-Pruning via Optimal Transport" 논문의 구현입니다. 기존의 Pruning 방식과 달리, Optimal Transport를 활용한 Intra-Fusion 방식을 통해 뉴런을 융합하여 네트워크의 표현력을 유지하면서 모델을 경량화합니다.

### Key Features
- Optimal Transport를 활용한 Intra-Fusion Pruning 구현
- 다양한 Pruning 방식 비교 (Default vs Intra-Fusion)
- 상세한 성능 분석 (정확도, 파라미터 수, 추론 시간)
- 다양한 스파시티(sparsity)에서의 실험 지원

## Project Structure
```
├── main.py                 # Main execution file
├── parameters.py           # Command line arguments
├── prune.py               # Pruning implementations
├── utils.py               # Utility functions
├── compare_models.py      # Model comparison tools
├── model_architectures/   # Model architecture definitions
│   ├── __init__.py
│   ├── vgg.py
│   └── resnet.py
├── logs/                  # Experiment logs
└── pruned_models/        # Saved model checkpoints
```

## Installation & Requirements

### Prerequisites
```bash
torch >= 1.7.0
torchvision
numpy
tqdm
thop           # For FLOPs calculation
```

### Setup
```bash
git clone https://github.com/username/intra-fusion.git
cd intra-fusion
pip install -r requirements.txt
```

## Usage

### Basic Training & Pruning
```bash
python main.py --model-name vgg11_bn --sparsities 0.3 0.5 0.7
```

### Arguments
- `--model-name`: 모델 아키텍처 선택 (vgg11_bn|resnet18)
- `--group_idxs`: Pruning할 그룹 지정 [1, 2, 3]
- `--sparsities`: Target sparsity 값 [0.3, 0.4, 0.5, 0.6, 0.7]
- `--importance_criteria`: 뉴런 중요도 측정 방식 (l1|taylor|lamp|chip)
- `--gpu_id`: GPU 설정 (-1: CPU)

### Model Comparison
```bash
python compare_models.py --model_dir ./pruned_models
```

## Implementation Details

### Pruning Methods

1. **Default Pruning**
```python
prune(
    model,
    importance_criteria="l1",
    sparsity=0.5,
    dimensionality_preserving=False
)
```

2. **Intra-Fusion Pruning**
```python
prune(
    model,
    importance_criteria="l1",
    sparsity=0.5,
    optimal_transport=ot,
    dimensionality_preserving=False
)
```

### Optimal Transport Configuration
```python
ot = OptimalTransport(
    p=1,                           # p-norm 값
    target_probability="uniform",  # 타겟 확률 분포
    source_probability="uniform",  # 소스 확률 분포
    target="most_important"       # 타겟 선택 방식
)
```

## Results Format

실험 결과는 JSON 형식으로 저장되며 다음 정보를 포함합니다:

```json
{
    "timestamp": "2024-01-17 10:00:00",
    "group_idx": 1,
    "sparsity": 0.5,
    "original_model": {
        "parameters": 1000000,
        "accuracy": 95.5,
        "inference_time": 10.5
    },
    "default_pruning": {
        "parameters": 500000,
        "accuracy": 94.0,
        "inference_time": 6.2
    },
    "intra_fusion": {
        "parameters": 500000,
        "accuracy": 95.0,
        "inference_time": 6.0
    }
}
```

## Citation

```bibtex
@inproceedings{theus2024metapruning,
  title={Towards Meta-Pruning via Optimal Transport},
  author={Alexander Theus and Olin Geimer and Friedrich Wicke and Thomas Hofmann and Sotiris Anagnostidis and Sidak Pal Singh},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2024},
}
```