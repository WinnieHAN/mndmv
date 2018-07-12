# lv_dmv_parser
Dependency parser using DMV model with latent variables
Parameters with best performance using vanilla DMV:
--train
data/wsj10_tr
--dev
data/wsj10_d
--epoch
25
--split_epoch
4
--param_smoothing
0.1
--split_factor
2
--cvalency
1
--em_type
viterbi
--batch
1000
--sub_batch
1000
--do_eval

Parameters with best performance using split-DMV(viterbi):
--train
data/wsj10_tr
--dev
data/wsj10_d
--epoch
25
--do_eval
--use_lex
--split_epoch
4
--param_smoothing
0.1
--split_factor
2
--cvalency
1
--do_split
--em_type
viterbi

Parameters with best performance using split-DMV(lateen):
--train
data/wsj10_tr
--dev
data/wsj10_d
--epoch
25
--do_eval
--use_lex
--split_epoch
4
--param_smoothing
0.1
--split_factor
2
--cvalency
2
--do_split
--em_type
viterbi
--em_after_split


