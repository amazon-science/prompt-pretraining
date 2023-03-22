## Pre-trained POMP Prompt
| Name  (configs)                                                                                                                    | ImageNet-21K Acc. | Cross-dataset Acc. | Cross-domain Acc. | Epochs | Prompt Length |                                                                  Model                                                                   |
|------------------------------------------------------------------------------------------------------------------------------------|:-----------------:|:------------------:|:-----------------:|:------:|:-------------:|:----------------------------------------------------------------------------------------------------------------------------------------:|
| [vit_b16_ep5_randaug2_unc1000_16shots_nctx4_cscFalse_ctpend_seed42.pth.tar](../configs/trainers/POMP/vit_b16_ep5_randaug2.yaml)    |       24.9        |        66.7        |       60.4        |   5    |       4       |                      [link](https://drive.google.com/file/d/1clOumlKZOCYwDGtY5WeCmIFFcy9YREJz/view?usp=share_link)                       | 
| [vit_b16_ep20_randaug2_unc1000_16shots_nctx16_cscFalse_ctpend_seed42.pth.tar](../configs/trainers/POMP/vit_b16_ep20_randaug2.yaml) |       25.2        |        65.1        |       60.0        |   20   |      16       |                      [link](https://drive.google.com/file/d/1C8oU6cWkJdU3Q3IHaqTcbIToRLo9bMnu/view?usp=share_link)                       |

make a `pretrained/` directory under the main directory `pomp/` and then download the above checkpoints to the `pretrained` directory. The directory structure should look like:
```
pomp/
|–– pretrained/
|   |–– vit_b16_ep5_randaug2_unc1000_16shots_nctx4_cscFalse_ctpend_seed42.pth.tar
|   |–– vit_b16_ep20_randaug2_unc1000_16shots_nctx16_cscFalse_ctpend_seed42.pth.tar
```

## POMP for object detection (based on Detic)
(1) Open-vocabulary LVIS

| Name  (configs)                                                                                                                                      | mAPr | mAP  | Prompt Legth |                                            Model                                              |
|------------------------------------------------------------------------------------------------------------------------------------------------------|:----:|:----:|:------------:|:---------------------------------------------------------------------------------------------:|
| [Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.pth](../third_party/Detic/configs/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.yaml) | 25.2 | 32.7 |      16      | [link](https://drive.google.com/file/d/1RURj-YqdwBy4QXSqVLrkESBVnmT2K_R0/view?usp=share_link) | 

(2) Cross-dataset

| Name  (configs)                                                                                                                              | AP50 on LVIS (source) | AP50 on COCO (target) | AP50 on Object365 (target) | Prompt Legth |                                            Model                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:---------------------:|:--------------------------:|:------------:|:---------------------------------------------------------------------------------------------:|
| [Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.pth](../third_party/Detic/configs/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.yaml) |         50.7          |         58.0          |            23.4            |      16      | [link](https://drive.google.com/file/d/1TwrjcUYimkI_f9z9UZXCmLztdgv31Peu/view?usp=share_link) | 

## POMP for semantic segmentation (based on ZSSeg)

(1) Open-vocabulary COCO Stuff

| Name  (configs)                                                                                                                                                                                                     | hIoU | mIoU-unseen | Prompt Legth |                                           Model                                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----:|:-----------:|:------------:|:---------------------------------------------------------------------------------------------:|
| [coco-stuff-164k-156_zero_shot_proposal_classification_learn_prompt_pomp_bs32_10k.pth](../third_party/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_proposal_classification_learn_prompt_pomp_bs32_10k.yaml) | 39.1 |    38.2     |      16      | [link](https://drive.google.com/file/d/1GlkoGATrh9jIq2WTlhTePt1BpoHZEhcG/view?usp=share_link) | 
| [coco-stuff-164k-156_zero_shot_maskformer_R101c_pomp_tuned_bs32_60k.pth](../third_party/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_pomp_prompt_bs32_60k.yaml)                            | 39.1 |    38.2     |      16      | [link](https://drive.google.com/file/d/1kNSxRfiewjAkRQ7KBHguz_mS-zDcwq7d/view?usp=share_link) | 

(2) Open-vocabulary Pascal VOC

| Name  (configs)                                                                                                                                                                                   | hIoU | mIoU-unseen | Prompt Legth |                                            Model                                              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----:|:-----------:|:------------:|:---------------------------------------------------------------------------------------------:|
| [voc-11k-15_zero_shot_proposal_classification_learn_prompt_pomp_bs16_10k.pth](../third_party/zsseg.baseline/configs/voc-11k-15/zero_shot_proposal_classification_learn_prompt_pomp_bs16_10k.yaml) | 84.4 |    76.8     |      16      | [link](https://drive.google.com/file/d/1MTHCWl20_fJ8WAOfRvsKzdJnqHaQe2GS/view?usp=share_link) | 
| [voc-11k-15_zero_shot_maskformer_R101c_pomp_tuned_bs16_20k.pth](../third_party/zsseg.baseline/configs/voc-11k-15/zero_shot_maskformer_R101c_pomp_prompt_bs16_20k.yaml)                            | 84.4 |    76.8     |      16      | [link](https://drive.google.com/file/d/1MfMcFDtxwbnnx5Hk9W8NQuGOg448VF7s/view?usp=share_link) | 

(3) Cross-dataset

| Name  (configs)                                                                                                                                                                          | mIoU on COCO Stuff (source) | mIoU on ADE20K (target) | mIoU on PASCAL Context (target) | Prompt Legth |                                           Model                                               |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------:|:-----------------------:|:-------------------------------:|:------------:|:---------------------------------------------------------------------------------------------:|
| [coco-stuff-164k-171_zero_shot_maskformer_R101c_pomp_tuned_bs32_60k.pth](../third_party/zsseg.baseline/configs/coco-stuff-164k-171/zero_shot_maskformer_R101c_pomp_prompt_bs32_60k.yaml) |            41.1             |          20.7           |              51.1               |      16      | [link](https://drive.google.com/file/d/1byR4DG9iAwyN5y2rYtT-dow1ZaoAhN1U/view?usp=share_link) | 
