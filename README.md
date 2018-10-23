# DONE: line 142  (ml_dmv_parser.py )   batch_predict_lan_v should be manually  given a value. I will revise it later.

# DONE: line 308 (def evaluate_batch_score of ml_dmv_model.py) no use neural:
# if self.initial_flag: ---> if True:

if set non_neural_iter>epochs, then it is DMV model.

python src/ml_dmv_parser.py --train en-fr --dev en --child_neural --em_type em --cvalency 2 --do_eval --ml_comb_type 1 --bidirectional --child_only --em_iter 1 --function_mask

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


