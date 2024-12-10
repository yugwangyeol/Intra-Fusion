# Intra-Fusion

![image](https://github.com/user-attachments/assets/a7ebb6ce-8aac-4692-b7ce-65fcbebe8c0e)

Towards Meta-Pruning via Optimal Transport는 2024년 2월에 발표된 논문으로 기존의 pruning에서 Intra-Fusion을 추가한 논문이다. 간단히 요약하자면 기존의 pruning 방식은 중요하지 않은 뉴런을 제거하고, 네트워크 성능을 유지하기 위해 추가적인 미세 조정을 필요로 했다면, Intra-Fusion은 Optimal Transport 기법을 사용하여 뉴런을 단순히 제거하는 대신, 덜 중요한 뉴런을 남아있는 뉴런에 fuse하여 네트워크의 표현력을 유지하여 뛰어난 성능을 보였다. 해당 모델에 대해 정확도, Parameter size, Inference time을 측정하고 비교하는 작업 진행하였다.

## Citation
```
@inproceedings{theus2024metapruning,
  title={Towards Meta-Pruning via Optimal Transport},
  author={Alexander Theus and Olin Geimer and Friedrich Wicke and Thomas Hofmann and Sotiris Anagnostidis and Sidak Pal Singh},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2024},
}
```