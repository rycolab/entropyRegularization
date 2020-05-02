## Generalized Entropy Regularization

This library is built on top of [fairseq (pytorch)](https://github.com/pytorch/fairseq).

Generalized entropy regularization can be used with any probabilistic model and data set. Just set the `--criterion` flag to `jensen_cross_entropy` and specify `--alpha` and `--beta` when running `fairseq-train`. Specify `--use-uniform` to use the uniform distribution as the baseline. Otherwise, the unigram distribution with annealing parameter `--T` will be used (run `fairseq-train --help` to see all options).

## Examples

### Neural Machine Translation
Preprocess data following examples in the [Translation README](examples/translation/README.md). A convolutional model can then be trained on IWSLT'14 (De-En) with the following command:

```
fairseq-train data-bin/iwslt14.tokenized.de-en \
			--arch fconv_iwslt_de_en --max-tokens 4000 --update-freq 8 \
            --clip-norm 0.1 --dropout 0.2 --criterion jensen_cross_entropy \
            --lr-scheduler fixed --min-lr 1e-8 --lr 0.5 --alpha 0.5 \
            --beta 0.7 --use-uniform
```

Likewise, a Transformer can be trained as follows:

```
fairseq-train data-bin/iwslt14.tokenized.de-en \
		    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		    --dropout 0.3 --weight-decay 0.0001 --max-tokens 4000 \
		    --criterion jensen_cross_entropy  --alpha 0.5 \
            --beta 0.7 --use-uniform
```

Generation is the same as in the Fairseq Translation README.

### Abstractive Summarization

Download the CNN/DailyMail data set according to the [BART README](examples/bart/README.md). Follow their suggested training method, setting `--criterion` to `jensen_cross_entropy` and specifying `--alpha`,  `--beta`, and `--use-uniform` (if desired).


Other models can be trained using the same methodology as above. See [fairseq documentation](https://fairseq.readthedocs.io/) for more options.

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options

Installation:
```bash
git clone https://github.com/rycolab/entropyRegularization
cd entropy_regularization
pip install --editable .
```



# Citation

This code is for the paper _Generalized Entropy Regularization or: There’s Nothing Special about Label Smoothing_ featured in ACL 2020. Please cite as:

```bibtex
@inproceedings{meister+al.acl20,
 title = {Generalized Entropy Regularization or: {T}here's Nothing Special about Label Smoothing},
 author = {Meister, Clara and 
Salesky, Elizabeth and 
Cotterell, Ryan},
 booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
 month = {July},
 year = {2020},
 address = {Seattle, USA},
 publisher = {Association for Computational Linguistics},
}
```

Feel free to include the fairseq citation as well; they're awesome.

```bibtex
@inproceedings{ott-etal-2019-fairseq,
    title = "fairseq: {A} Fast, Extensible Toolkit for Sequence Modeling",
    author = "Ott, Myle  and
      Edunov, Sergey  and
      Baevski, Alexei  and
      Fan, Angela  and
      Gross, Sam  and
      Ng, Nathan  and
      Grangier, David  and
      Auli, Michael",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics (Demonstrations)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-4009",
    doi = "10.18653/v1/N19-4009",
    pages = "48--53",
}
```
