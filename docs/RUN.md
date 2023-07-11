# Pre-training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each prompting variant including POMP, CoOp, CoCoOp, MaPLe, and VPT.
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `pomp/`.
Below we provide training and evaluation instructions for POMP. 

### Pre-training time and compute
We pre-train POMP on ImageNet-21K with a batch size of 32 using **8** NVIDIA V100 GPU.
Pre-training POMP on ImageNet-21K for 5 epochs takes ~1 hours.

## POMP for image classification
Run our demo using Colab: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1OEFw1GfKXogx8mdFS2pClLjPK3aPEZyY?usp=sharing)

(1) Pre-training on ImageNet-21K
```bash
# dataset=imagenet_21k, backbone=vit_b16_ep5_randaug2, class-token-position=end, prompt-length=4, #shot=16, class-specifc-prompt=False, K=1000
sh scripts/pomp/main.sh imagenet_21k vit_b16_ep5_randaug2 end 4 16 False 1000
# evaluate on valid set and find the best checkpoint (need to modify the arguments in validation_test.py)
python validation_test.py
```

(2) Cross-dataset evaluation on 14 image classification downstream datasets
```bash
sh scripts/pomp/xd_test.sh
# display all results
ls output/evaluation/POMP/vit_b16_ep5_randaug2_unc1000_16shots_nctx4_best_val_*/seed42/log.txt | xargs -I {} sh -c 'echo {}; cat {} | grep accuracy'
```

## POMP for object detection (based on Detic)
Run our demo using Jupyter (you need to [prepare Detic environment](../third_party/Detic/docs/INSTALL.md)): [demo.ipynb](../third_party/Detic/demo/demo.ipynb)

Please first [prepare Detic environment](../third_party/Detic/docs/INSTALL.md), [prepare datasets](../third_party/Detic/datasets/README.md), [prepare ImageNet-21K pretrained backbone](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md#third-party-imagenet-21k-pretrained-models) and [prepare pre-trained POMP prompt](MODELS.md).

Run the following commands from the directory `prompt-pretraining/third_party/Detic`.

(1) Create class vectors based on pre-trained POMP prompt
```bash
# for LVIS
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --prompt none --out_path datasets/metadata/lvis_v1_clip_pomp+cname.npy --model POMP --soft_prompt ~/prompt-pretraining/pretrained/vit_b16_ep20_randaug2_unc1000_16shots_nctx16_cscFalse_ctpend_seed42.pth.tar
# for COCO
python tools/dump_clip_features.py --ann datasets/coco/zero-shot/instances_val2017_all_2_oriorder.json --prompt none --out_path datasets/metadata/coco_clip_pomp+cname.npy --model POMP --soft_prompt ~/prompt-pretraining/pretrained/vit_b16_ep20_randaug2_unc1000_16shots_nctx16_cscFalse_ctpend_seed42.pth.tar
# for Object365
python tools/dump_clip_features.py --ann datasets/objects365/annotations/zhiyuan_objv2_val_fixname.json --prompt none --out_path datasets/metadata/o365_fixname_clip_pomp+cname.npy --model POMP --soft_prompt ~/prompt-pretraining/pretrained/vit_b16_ep20_randaug2_unc1000_16shots_nctx16_cscFalse_ctpend_seed42.pth.tar
```

(2) Open-vocabulary LVIS
```bash
# pre-train object proposal network on LVIS-base
python train_net.py --num-gpus 8 --config-file configs/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x_pomp.yaml
python train_net.py --num-gpus 8 --config-file configs/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.yaml

# evaluation
python train_net.py --num-gpus 8 --config-file configs/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.yaml --eval-only MODEL.WEIGHTS models/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp/model_final.pth
```

(3) Cross-dataset transfer
```bash
# pre-train object proposal network on standard LVIS
python train_net.py --num-gpus 8 --config-file configs/BoxSup-C2_L_CLIP_R5021k_640b64_4x_pomp.yaml
python train_net.py --num-gpus 8 --config-file configs/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.yaml

# evaluation
python train_net.py --num-gpus 8 --config-file configs/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp_cross_datasets.yaml --eval-only
```

## POMP for semantic segmentation (based on ZSSeg)
Please first [prepare ZSSeg environment and datasets](../third_party/zsseg.baseline/README.md#Guideline), and [prepare pre-trained POMP prompt](MODELS.md).

Run the following commands from the directory `prompt-pretraining/third_party/zsseg.baseline`.

(1) Open-vocabulary COCO Stuff
```bash
python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_proposal_classification_learn_prompt_pomp_bs32_10k.yaml --num-gpus 8 
python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_pomp_prompt_bs32_60k.yaml --num-gpus 8
```

(2) Open-vocabulary Pascal VOC
```bash
python train_net.py --config-file configs/voc-11k-15/zero_shot_proposal_classification_learn_prompt_pomp_bs16_10k.yaml --num-gpus 8 
python train_net.py --config-file configs/voc-11k-15/zero_shot_maskformer_R101c_pomp_prompt_bs16_20k.yaml --num-gpus 8
```

(3) Cross-dataset transfer
```bash
# pre-train mask proposal network on standard COCO Stuff
python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_pomp_prompt_bs32_60k.yaml --num-gpus 8

# evaluation
# for ADE20K
python train_net.py --eval-only --resume --config-file configs/ade20k-150/cross_dataset_pomp_prompt_test_only.yaml --num-gpus 8 MODEL.WEIGHTS output/coco-stuff-164k-171/zero_shot_maskformer_R101c_pomp_prompt_bs32_60k/model_final.pth
# for PASCAL Context
python train_net.py --eval-only --resume --config-file configs/pcontext-59/cross_dataset_pomp_prompt_test_only.yaml --num-gpus 8 MODEL.WEIGHTS output/coco-stuff-164k-171/zero_shot_maskformer_R101c_pomp_prompt_bs32_60k/model_final.pth
```