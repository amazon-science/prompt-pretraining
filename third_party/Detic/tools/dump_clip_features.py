# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import torch
import numpy as np
import itertools
from nltk.corpus import wordnet
import sys
import torch.distributed as dist
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--soft_prompt', default='/home/ubuntu/efs/detpro/checkpoints/DetPro-mask-rcnn/iou_neg5_ens.pth')
    parser.add_argument('--out_path', default='')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--fix_space', action='store_true')
    parser.add_argument('--use_underscore', action='store_true')
    parser.add_argument('--avg_synonyms', action='store_true')
    parser.add_argument('--use_wn_name', action='store_true')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in \
        sorted(data['categories'], key=lambda x: x['id'])]
    if 'synonyms' in data['categories'][0]:
        if args.use_wn_name:
            synonyms = [
                [xx.name() for xx in wordnet.synset(x['synset']).lemmas()] \
                    if x['synset'] != 'stop_sign.n.01' else ['stop_sign'] \
                    for x in sorted(data['categories'], key=lambda x: x['id'])]
        else:
            synonyms = [x['synonyms'] for x in \
                sorted(data['categories'], key=lambda x: x['id'])]
    else:
        synonyms = []
    if args.fix_space:
        cat_names = [x.replace('_', ' ') for x in cat_names]
    if args.use_underscore:
        cat_names = [x.strip().replace('/ ', '/').replace(' ', '_') for x in cat_names]
    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
            for x in synonyms]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
            for x in synonyms]

    print('sentences_synonyms', len(sentences_synonyms), \
        sum(len(x) for x in sentences_synonyms))
    if args.model == 'clip':
        import clip
        print('Loading CLIP')
        model, preprocess = clip.load(args.clip_model, device=device)
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        text = clip.tokenize(sentences).to(device)
        with torch.no_grad():
            if len(text) > 10000:
                text_features = torch.cat([
                    model.encode_text(text[:len(text) // 2]),
                    model.encode_text(text[len(text) // 2:])],
                    dim=0)
            else:
                text_features = model.encode_text(text)
        print('text_features.shape', text_features.shape)
        if args.avg_synonyms:
            synonyms_per_cat = [len(x) for x in sentences_synonyms]
            text_features = text_features.split(synonyms_per_cat, dim=0)
            text_features = [x.mean(dim=0) for x in text_features]
            text_features = torch.stack(text_features, dim=0)
            print('after stack', text_features.shape)
        text_features = text_features.cpu().numpy()
    elif args.model in ['bert', 'roberta']:
        from transformers import AutoTokenizer, AutoModel
        if args.model == 'bert':
            model_name = 'bert-large-uncased' 
        if args.model == 'roberta':
            model_name = 'roberta-large' 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        if args.avg_synonyms:
            sentences = list(itertools.chain.from_iterable(sentences_synonyms))
            print('flattened_sentences', len(sentences))
        inputs = tokenizer(sentences, padding=True, return_tensors="pt")
        with torch.no_grad():
            model_outputs = model(**inputs)
            outputs = model_outputs.pooler_output
        text_features = outputs.detach().cpu()
        if args.avg_synonyms:
            synonyms_per_cat = [len(x) for x in sentences_synonyms]
            text_features = text_features.split(synonyms_per_cat, dim=0)
            text_features = [x.mean(dim=0) for x in text_features]
            text_features = torch.stack(text_features, dim=0)
            print('after stack', text_features.shape)
        text_features = text_features.numpy()
        print('text_features.shape', text_features.shape)
    elif args.model == 'detpro':
        text_features = torch.load(args.soft_prompt).cpu().numpy()
    elif args.model == 'POMP' or args.model == 'CoOp':
        try:
            ckpt = torch.load(args.soft_prompt)
        except:
            map_location = 'cuda:%d' % dist.get_rank() if torch.cuda.is_available() and dist.is_initialized() else 'cpu'
            ckpt = torch.load(args.soft_prompt, map_location=map_location)
        ctx = ckpt["state_dict"]['ctx'].to(device)
        n_ctx = ctx.size(0)

        import clip
        print('Loading CLIP')
        model, preprocess = clip.load('ViT-B/16', device=device)
        text_encoder = TextEncoder(model)

        dtype = model.dtype
        classnames = [name.replace("_", " ") for name in cat_names]
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).int().to(device)

        with torch.no_grad():
            embedding = model.token_embedding(tokenized_prompts).type(dtype)
            prefix = embedding[:, :1, :]
            suffix = embedding[:, 1 + n_ctx:, :]
            ctx = ctx.unsqueeze(0).expand(len(cat_names), -1, -1)

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            text_features = text_encoder(prompts, tokenized_prompts).cpu().numpy()
    else:
        assert 0, args.model
    if args.out_path != '':
        print('saveing to', args.out_path)
        np.save(open(args.out_path, 'wb'), text_features)
