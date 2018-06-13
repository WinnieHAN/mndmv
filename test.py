import os
import pickle
from optparse import OptionParser

import numpy as np
import torch

import dmv_model
import eisner_for_dmv
import utils

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--test", dest="test", help="test file", metavar="FILE", default="data/wsj10_te")

    parser.add_option("--batch", type="int", dest="batchsize", default=100)

    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="output/dmv.model")

    parser.add_option("--outdir", type="string", dest="output", default="output")

    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)

    parser.add_option("--paramem", dest="paramem", help="EM parameters file", metavar="FILE",
                      default="paramem.pickle")

    parser.add_option("--log", dest="log", help="log file", metavar="FILE", default="output/log")

    parser.add_option("--idx", type="int", dest="model_idx", default=1)

    parser.add_option("--sample_idx", type="int", dest="sample_idx", default=1000)

    parser.add_option("--gpu", type="int", dest="gpu", default=-1, help='gpu id, set to -1 if use cpu mode')

    (options, args) = parser.parse_args()

    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        print 'To use gpu' + str(options.gpu)

    with open(os.path.join(options.output, options.params + '_' + str(options.sample_idx)), 'r') as paramsfp:
        w2i, pos, stored_opt = pickle.load(paramsfp)
        # stored_opt.external_embedding = options.external_embedding
    sentences = utils.read_data(options.test, True)
    learned_model = dmv_model.ldmv_model(w2i, pos, stored_opt)
    learned_model.load(options.model + str(options.model_idx) + '_' + str(options.sample_idx))

    with open(os.path.join(options.output, options.paramem + str(options.model_idx) + '_' + str(options.sample_idx)),
              'r') as paramem:
        learned_model.trans_param, learned_model.decision_param, learned_model.lex_param, learned_model.tag_num = pickle.load(
            paramem)

    print 'Model loaded'
    learned_model.eval()
    outpath = os.path.join(options.output, 'test_pred' + '_' + str(options.sample_idx))
    eval_sentences = utils.read_data(options.test, True)
    learned_model.eval()
    eval_sen_idx = 0
    eval_data_list = list()

    for s in eval_sentences:
        s_word, s_pos = s.set_data_list(w2i, pos)
        s_data_list = list()
        s_data_list.append(s_word)
        s_data_list.append(s_pos)
        s_data_list.append([eval_sen_idx])
        eval_data_list.append(s_data_list)
        eval_sen_idx += 1
    eval_batch_data = utils.construct_batch_data(eval_data_list, options.batchsize)
    parse_results = {}
    for batch_id, one_batch in enumerate(eval_batch_data):
        eval_batch_words, eval_batch_pos, eval_batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], \
                                                           [s[2][0] for s in one_batch]
        eval_batch_words = np.array(eval_batch_words)
        eval_batch_pos = np.array(eval_batch_pos)
        batch_score, batch_decision_score = learned_model.evaluate_batch_score(eval_batch_words, eval_batch_pos)
        batch_parse = eisner_for_dmv.batch_parse(batch_score, batch_decision_score, learned_model.dvalency,
                                                 learned_model.cvalency)
        for i in range(len(eval_batch_pos)):
            parse_results[eval_batch_sen[i]] = (batch_parse[0][i], batch_parse[1][i])
    utils.eval(parse_results, eval_sentences, outpath,
               options.log + str(options.model_idx) + '_' + str(options.sample_idx), 0)
