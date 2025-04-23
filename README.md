# Transformer-Based Printing Quality Recognition in Fused Filament Fabrication

This repository contains PyTorch evaluation code, training code and pretrained models for the following papers:
**"Transformer-based approach for printing quality recognition in fused filament fabrication"**  

**Abstract:**  
Ensuring high-quality prints in additive manufacturing is a critical challenge due to the variability in materials, process parameters, and equipment. Machine learning models are increasingly being employed for real-time quality monitoring, enabling the detection and classification of defects such as under-extrusion and over-extrusion. Vision Transformers (ViTs), with their global self-attention mechanisms, offer a promising alternative to traditional convolutional neural networks (CNNs). This paper presents a transformer-based approach for print quality recognition in additive manufacturing technologies, with a focus on fused filament fabrication (FFF), leveraging advanced self-supervised representation learning techniques to enhance the robustness and generalizability of ViTs. Here, we show that the ViT model effectively classifies printing quality into different levels of extrusion, achieving exceptional performance across varying dataset scales and noise levels. Training evaluations show a steady decrease in cross-entropy loss, with prediction accuracy, precision, recall, and the harmonic mean of precision and recall (F1) scores reaching close to 1 within 40 epochs, demonstrating excellent performance across all classes. The macro and micro F1 scores further emphasize the ability of ViT to handle both class imbalance and instance-level accuracy effectively. Our results also demonstrate that ViT outperforms CNN in all scenarios, particularly in noisy conditions and with small datasets. Comparative analysis reveals ViT’s advantages, particularly in leveraging global self-attention and robust feature extraction methods, which enhance its ability to generalize effectively and remain resilient with limited data. These findings underline the potential of the transformer-based approach as a scalable, interpretable, and reliable solution to real-time quality monitoring in FFF, addressing key challenges in additive manufacturing defect detection and ensuring process efficiency.

## 📑 Paper
If you find this work useful, please cite our paper:
@article{wang2025transformer,
  title={Transformer-based approach for printing quality recognition in fused filament fabrication},
  author={Wang, Xing Quan and Jin, Zeqing and Zheng, Bowen and Gu, Grace X},
  journal={npj Advanced Manufacturing},
  volume={2},
  number={1},
  pages={1--9},
  year={2025},
  publisher={Nature Publishing Group}
}
```bibtex

</details>
If you find this repository useful, please consider giving a star ⭐ and cite the relevant papers. 

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
