<p align="center">
  <img src="asset/caption.png" alt="SparseVideoNav Logo">
</p>

<p align="center">
<h1 align="center"><strong>SparseVideoNav: Sparse Video Generation Propels Real-World Beyond-the-View Vision-Language Navigation</strong></h1>
<p align="center">
    <a href='https://betray12138.github.io/resume/' target='_blank'>Hai Zhang</a><sup>*</sup>&emsp;
    <a href='https://github.com/stdcat' target='_blank'>Siqi Liang</a><sup>*</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en' target='_blank'>Li Chen</a>&emsp;
    <a href='https://github.com/LiYuxian0306' target='_blank'>Yuxian Li</a>&emsp;
    <a href='https://github.com/PUFFERFISHTNT' target='_blank'>Yukuan Xu</a>&emsp;
    <a href='https://z-taylcr7.github.io/' target='_blank'>Yichao Zhong</a>&emsp;
    <a href='https://mars.hku.hk/people.html' target='_blank'>Fu Zhang</a><sup>†</sup>&emsp;
    <a href='https://lihongyang.info/' target='_blank'>Hongyang Li</a><sup>†</sup>&emsp;
    <br><br>
    <a href='https://hku.hk/' target='_blank'><img src="asset/HKU_logo.png" alt="Project Page" height="40"></a>
    <br>
    The University of Hong Kong&emsp;
    <br>
  </p>
</p>

<div id="top" align="center">

<div>
<a href="https://opendrivelab.com/SparseVideoNav" target="_blank"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://github.com/OpenDriveLab/SparseVideoNav"><img alt="Repo" src="https://img.shields.io/badge/github-repo-blue?logo=github"/></a>
<a href="https://arxiv.org/abs/2602.05827" target="_blank"><img src="https://img.shields.io/badge/arXiv-2602.05827-b31b1b" alt="arXiv"></a>
<a href="https://creativecommons.org/licenses/by-nc-sa/4.0" target="_blank"><img src="https://img.shields.io/badge/License-CC_%20_BY--NC--SA_4.0-blue.svg" alt="License"/></a>
</div>


</div>

## 📖 Introduction 

SparseVideoNav introduces video generation models to real-world beyond-the-view vision-language navigation for the first time. It achieves sub-second trajectory inference with a sparse future spanning a 20-second horizon, yielding a remarkable 27× speed-up. Real-world zero-shot experiments show 2.5× higher success rate than state-of-the-art LLM baselines and mark the first realization in challenging night scenes.

**Developers**: [Hai Zhang](https://zhanghenryhai12138.github.io/) and [Siqi Liang](https://github.com/stdcat)

## 📢 News

> [!IMPORTANT]
> 🌟 Stay up to date at [opendrivelab.com](https://opendrivelab.com/#news)!

- 🎉 **2026-02-05**: [Project Page](https://opendrivelab.com/SparseVideoNav) is now available!
- 🎉 **2026-02-06**: [arXiv preprint](https://arxiv.org/abs/2602.05827) is now available!

## 📬 Contact

For further inquiries or assistance, please contact [zhanghenryhai12138@gmail.com](mailto:zhanghenryhai12138@gmail.com) or [liangsiqi@connect.hku.hk](mailto:liangsiqi@connect.hku.hk)

## 📌 Table of Contents
- 📖 [Introduction](#-introduction)
- 📢 [News](#-news)
- 📬 [Contact](#-contact)
- 🔥 [Highlights](#highlights)
- 📝 [TODO List](#-todo-list)
- 📄 [License and Citation](#-license-and-citation)


## 🔥Highlights

- We investigate beyond-the-view navigation tasks in the real world  by introducing video generation model to this field for the first time.
- We pioneer a paradigm shift from continuous to sparse video generation for longer prediction horizon.
- We achieve sub-second trajectory inference guided by a generated sparse future spanning a 20-second horizon. This yields a remarkable 27x speed-up compared to the unoptimized counterpart.
- We achieve the first realization of beyond-the-view navigation in challenging night scenes with a 17.5% success rate.

## 📝 TODO List
- [x] SparseVideoNav Paper Release.
  - [x] arXiv preprint is now available! 
- [ ] SparseVideoNav Code Release.
  - [ ] Inference code of distilled video generation model and model checkpoint (Estimate 2026 March).
  - [ ] Inference code of continuous action head and model checkpoint (Estimate 2026 Q3).
- [ ] SparseVideoNav Dataset Release
  - [ ] ~140h real-world VLN data (Estimate 2026 Q3).


## 📄 License and Citation

All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

- Please consider citing our work if it helps your research.
```BibTeX
@article{zhang2026sparse,
  title={Sparse Video Generation Propels Real-World Beyond-the-View Vision-Language Navigation},
  author={Zhang, Hai and Liang, Siqi and Chen, Li and Li, Yuxian and Xu, Yukuan and Zhong, Yichao and Zhang, Fu and Li, Hongyang},
  journal={arXiv preprint arXiv:2602.05827},
  year={2026}
}
```
