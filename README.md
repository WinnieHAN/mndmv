# Multilingual Grammar Induction with Continuous Language Identification

This is PyTorch implementation of the [paper](http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/emnlp19mult.pdf):
```
Multilingual Grammar Induction with Continuous Language Identification
Wenjuan Han, Ge Wang, Yong Jiang, Kewei Tu
EMNLP 2019
```

The code performs unsupervised grammar learning in the dataset of 15 languages selected from UD treebanks v1.4.

Please concact hanwj@shanghaitech.edu.cn or wangge@shanghaitech.edu.cn if you have any questions.

## Environments

- Python 2.7
- PyTorch >=1.0
- Numpy  1.15.4

## Data

run:
```shell
python ml_dmv_parser.py --cvalency 2 --do_eval --neural_epoch 1 --function_mask --lr 0.001 --epoch 70 --use_neural --embed_languages --em_type em --child_only --language_predict
```
The grammar induction is by default set to be performed on 15 selected languages, you can customize your own language selection by changing the contents in the language_list file


## Reference
```
@inproceedings{han2019multilingual,
  title={Multilingual grammar induction with continuous language identification},
  author={Han, Wenjuan and Wang, Ge and Jiang, Yong and Tu, Kewei},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={5728--5733},
  year={2019}
}
```

