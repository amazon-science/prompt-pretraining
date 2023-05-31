# Prompt Pre-Training with Over Twenty-Thousand Classes for Open-Vocabulary Visual Recognition

<h5 align="center"><i>"Scaling up prompt learning on ImageNet-21K achieves SOTA on 21 downstream datasets."</i></h5>

> [**Prompt Pre-Training with Over Twenty-Thousand Classes for Open-Vocabulary Visual Recognition**](https://arxiv.org/abs/2304.04704)<br>
> [Shuhuai Ren](https://renshuhuai-andy.github.io/), [Aston Zhang](https://www.astonzhang.com/), [Yi Zhu](https://bryanyzhu.github.io/), [Shuai Zhang](https://shuaizhang.tech/), [Shuai Zheng](https://szhengac.github.io/), [Mu Li](http://www.cs.cmu.edu/~muli/), [Alex Smola](https://alex.smola.org/), [Xu Sun](https://xusun.org/index.htm)


[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2304.04704) 
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1OEFw1GfKXogx8mdFS2pClLjPK3aPEZyY?usp=sharing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-pre-training-with-twenty-thousand/prompt-engineering-on-imagenet-21k)](https://paperswithcode.com/sota/prompt-engineering-on-imagenet-21k?p=prompt-pre-training-with-twenty-thousand) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-pre-training-with-twenty-thousand/prompt-engineering-on-imagenet-a)](https://paperswithcode.com/sota/prompt-engineering-on-imagenet-a?p=prompt-pre-training-with-twenty-thousand)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-pre-training-with-twenty-thousand/prompt-engineering-on-imagenet-r)](https://paperswithcode.com/sota/prompt-engineering-on-imagenet-r?p=prompt-pre-training-with-twenty-thousand)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-pre-training-with-twenty-thousand/prompt-engineering-on-imagenet-s)](https://paperswithcode.com/sota/prompt-engineering-on-imagenet-s?p=prompt-pre-training-with-twenty-thousand)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-pre-training-with-twenty-thousand/open-vocabulary-semantic-segmentation-on-coco)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-coco?p=prompt-pre-training-with-twenty-thousand)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-pre-training-with-twenty-thousand/open-vocabulary-semantic-segmentation-on-5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-5?p=prompt-pre-training-with-twenty-thousand) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/prompt-pre-training-with-twenty-thousand/open-vocabulary-object-detection-on-lvis-v1-0)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-lvis-v1-0?p=prompt-pre-training-with-twenty-thousand) 

# :rocket: News
* **(May 31, 2023)** 
  * Inference demo for image classification in Google Colab. 
* **(Mar 22, 2023)** 
  * Codes for prompt pretraining (POMP) on ImageNet-21K, cross-dataset and cross-task evaluation.
  * Checkpoints of pre-trained POMP prompts, segmentation backbones, and detection backbones.
<hr />

## Highlights

![main figure](docs/main_figure.png)


## Main Contributions

1) We introduce a prompt pre-training method POMP, which fisrt enables prompt learning on large-scale datasets like ImageNet-21K with over twenty-thousand classes.
2) POMP is memory and computation efficient. Compared with previous methods like CoOp, it achieves comparable accuracy on ImageNet-1K with only 19\% GPU memory and 50\% training time.
3) POMP achieves new SOTAs on various open-vocabulary visual recognition datasets and tasks.

## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data preparation
Please follow the instructions at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.

## Pre-trained Models
Please follow the instructions at [MODELS.md](docs/MODELS.md) to prepare all pre-trained models.

## Training and Evaluation
Please refer to the [RUN.md](docs/RUN.md) for detailed instructions on training, evaluating and reproducing the results.


<hr />

## Contact
If you have any questions, please feel free to create an issue on this repository.

## Citation
If you find this code useful for your research, please consider citing:
```
@article{ren2023pomp,
  title={Prompt Pre-Training with Twenty-Thousand Classes for Open-Vocabulary Visual Recognition},
  author={Ren, Shuhuai and Zhang, Aston and Zhu, Yi and Zhang, Shuai and Zheng, Shuai and Li, Mu and Smola, Alex and Sun, Xu},
  journal={arXiv preprint arXiv:2304.04704},
  year={2023}
}
```

## Acknowledgements

Our code is based on [CoOp](https://github.com/KaiyangZhou/CoOp), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch), [Detic](https://github.com/facebookresearch/Detic) and [ZSSeg](https://github.com/MendelXu/zsseg.baseline) repositories. We thank the authors for releasing their code. 
