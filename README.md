# Intra-Fusion

<div align="center">
  <img src="https://github.com/user-attachments/assets/a7ebb6ce-8aac-4692-b7ce-65fcbebe8c0e" alt="Intra-Fusion">
</div>

**Towards Meta-Pruning via Optimal Transport**는 2024년 2월에 발표된 논문으로, 기존의 Pruning 방식에 **Intra-Fusion** 개념을 추가한 논문이다.  
기존 Pruning 방식은 중요하지 않은 뉴런을 제거하고, 네트워크 성능을 유지하기 위해 추가적인 **미세 조정**이 필요했다.  

이에 반해, **Intra-Fusion**은 **Optimal Transport** 기법을 활용하여 뉴런을 단순히 제거하는 대신, 덜 중요한 뉴런을 남아있는 뉴런에 **융합(fuse)**하여 네트워크의 표현력을 유지한다.  
이를 통해 성능 저하 없이도 더 뛰어난 효율성을 달성할 수 있었다.  

해당 모델의 **정확도**, **Parameter Size**, **Inference Time** 등을 측정하고 기존 방법과 비교하는 실험을 진행하였다.

---

## Citation

```bibtex
@inproceedings{theus2024metapruning,
  title={Towards Meta-Pruning via Optimal Transport},
  author={Alexander Theus and Olin Geimer and Friedrich Wicke and Thomas Hofmann and Sotiris Anagnostidis and Sidak Pal Singh},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2024},
}
