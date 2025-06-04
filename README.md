<h1 align="center">Seg2Any: Open-set Segmentation-Mask-to-Image Generation with Precise Shape and Semantic Control</h1> 

<div align='center'>
    <a href="https://github.com/0xLDF" target="_blank">Danfeng Li</a><sup>1*</sup>,</span>
    <a href="https://huizhang0812.github.io/" target="_blank">Hui Zhang</a><sup>1*</sup>,</span>
    <a href="https://www.linkedin.com/in/sheng-wang-4620863a/" target="_blank">Sheng Wang</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=qkaJhBMAAAAJ&hl=zh-CN" target="_blank">Jiacheng Li<a><sup>2</sup>,
    <a href="https://zxwu.azurewebsites.net/" target="_blank">Zuxuan Wu</a><sup>1†</sup>
</div>

<div align='center'>
    <br><sup>1</sup>Fudan University <sup>2</sup>HiThink Research
    <br><small><sup>*</sup>Equal Contribution. <sup>†</sup>Corresponding author. </small>
</div>
<br>

<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://seg2any.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  <a href='https://arxiv.org/abs/2506.00596'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>
<br>

<p align="center">
  <img src="assets/demo.png" width="90%" height="90%">
</p>

## Overview

<p align="center">
  <img src="assets/framework_seg2any.png" width="90%" height="90%">
</p>

(a) An overview of the Seg2Any framework. Segmentation masks are transformed into Entity Contour Map, then encoded as condition tokens via frozen VAE. Negligible tokens are filtered out for efficiency. The resulting text, image, and condition tokens are concatenated into a unified sequence for MM-Attention. Our framework applies LoRA to all branches, achieving S2I generation with minimal extra parameters. (b) Attention Masks in MM-Attention, including Semantic Alignment Attention Mask and Attribute Isolation Attention Mask.

## Citation
If you find Seg2Any useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{li2025seg2any,
title={Seg2Any: Open-set Segmentation-Mask-to-Image Generation with Precise Shape and Semantic Control},
author={Li, Danfeng and Zhang, Hui and Wang, Sheng and Li, Jiacheng and Wu, Zuxuan},
journal={arXiv preprint arXiv:2506.00596},
year={2025}
}
```