import os
import pickle
import sys
from optparse import OptionParser

import numpy as np
import torch
from tqdm import tqdm

import eisner_for_dmv
import utils
from ml_dmv_model import ml_dmv_model as MLDMV
from ml_neural_m_step import m_step_model as MMODEL

# from torch_model.NN_module import *
import random

# from torch_model.NN_trainer import *

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train", help="train file", metavar="FILE", default="data/ud_file")
    parser.add_option("--dev", dest="dev", help="dev file", metavar="FILE", default="data/ud_file")

    parser.add_option("--batch", type="int", dest="batchsize", default=5000)
    parser.add_option("--sample_batch", type="int", dest="sample_batch_size", default=50000)

    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="output/dmv.model")

    parser.add_option("--pembedding", type="int", dest="pembedding_dim", default=5)
    parser.add_option("--epochs", type="int", dest="epochs", default=50)
    parser.add_option("--tag_num", type="int", dest="tag_num", default=1)

    parser.add_option("--dvalency", type="int", dest="d_valency", default=2)
    parser.add_option("--cvalency", type="int", dest="c_valency", default=1)
    parser.add_option("--em_type", type="string", dest="em_type", default='viterbi')

    parser.add_option("--count_smoothing", type="float", dest="count_smoothing", default=1e-8)
    parser.add_option("--param_smoothing", type="float", dest="param_smoothing", default=1e-8)

    parser.add_option("--optim", type="string", dest="optim_type", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.001)
    parser.add_option("--outdir", type="string", dest="output", default="output")

    parser.add_option("--sample_idx", type="int", dest="sample_idx", default=10000)
    parser.add_option("--do_eval", action="store_true", dest="do_eval", default=False)
    parser.add_option("--log", dest="log", help="log file", metavar="FILE", default="output/log")
    parser.add_option("--sub_batch", type="int", dest="sub_batch_size", default=5000)

    parser.add_option("--function_mask", action="store_true", default=False)

    parser.add_option("--e_pass", type="int", dest="e_pass", default=4)
    parser.add_option("--em_iter", type="int", dest="em_iter", default=1)

    parser.add_option("--paramem", dest="paramem", help="EM parameters file", metavar="FILE",
                      default="paramem.pickle")

    parser.add_option("--gpu", type="int", dest="gpu", default=-1, help='gpu id, set to -1 if use cpu mode')

    parser.add_option("--seed", type="int", dest="seed", default=0)
    parser.add_option("--drop_out", type="float", dest="drop_out", default=0.25)

    parser.add_option("--child_only", action="store_true", dest="child_only", default=False)
    parser.add_option("--valency_dim", type="int", dest="valency_dim", default=5)
    parser.add_option("--hid_dim", type="int", dest="hid_dim", default=10)
    parser.add_option("--pre_ouput_dim", type="int", dest="pre_output_dim", default=15)
    parser.add_option("--decision_pre_output_dim", type="int", dest="decision_pre_output_dim", default=5)
    parser.add_option("--neural_epoch", type="int", dest="neural_epoch", default=1)
    parser.add_option("--unified_network", action="store_true", dest="unified_network", default=False)
    parser.add_option("--reset_weight", action="store_true", dest="reset_weight", default=False)

    parser.add_option("--use_neural", action="store_true", dest="use_neural", default=False)
    parser.add_option("--sentence_predict", action="store_true", default=False)
    parser.add_option("--embed_languages", action="store_true", default=False)
    parser.add_option("--language_path", type="string", dest="language_path", default="language_list")
    parser.add_option("--non_neural_iter", type="int", dest="non_neural_iter", default=-1)
    parser.add_option("--lan_dim", type="int", dest="lan_dim", default=5)
    parser.add_option("--chosen_list", action="store_true", default=False)
    parser.add_option("--lstm_layer_num", type="int", dest="lstm_layer_num", default=1)
    parser.add_option("--lstm_hidden_dim", type="int", dest="lstm_hidden_dim", default=10)
    parser.add_option("--bidirectional", action="store_true", dest="bidirectional", default=False)
    parser.add_option("--eval_new_language", action="store_true", default=False)
    parser.add_option("--concat_all", action="store_true", default=False)
    parser.add_option("--language_predict",action="store_true",default=False)

    (options, args) = parser.parse_args()


    def do_eval(dmv_model, m_model, pos, languages, language_map, epoch, options):
        print "===================================="
        print 'Do evaluation'
        if not options.eval_new_language:
            eval_language_set = languages.keys()
            eval_languages = languages
        else:
            eval_language_set = utils.read_language_list(options.language_path)
            eval_languages = {l: i for i, l in enumerate(eval_language_set)}
        eval_file_list = os.listdir(options.dev)
        eval_file_set = utils.get_file_set(eval_file_list, eval_language_set, False)
        eval_sentences, eval_language_map = utils.read_multiple_data(options.dev, eval_file_set, True)
        dmv_model.eval()
        if options.use_neural:
            m_model.eval()
        devpath = os.path.join(options.output, 'eval_pred' + str(epoch + 1) + '_' + str(options.sample_idx))
        eval_data_list, _, eval_sentence_map = utils.construct_ml_pos_data(eval_sentences, pos, eval_languages,
                                                                           eval_language_map)
        eval_batch_data = utils.construct_batch_data(eval_data_list, options.batchsize)
        parse_results = {}
        classify_results = np.zeros(len(eval_data_list))
        if options.sentence_predict and epoch > options.non_neural_iter:
            eval_trans_param = np.zeros((len(eval_data_list), len(pos.keys()), len(pos.keys()), 2, options.c_valency))
        else:
            eval_trans_param = None
        for batch_id, one_batch in enumerate(eval_batch_data):
            eval_batch_pos, eval_batch_lan, eval_batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], [
                s[2][0] for s in one_batch]
            eval_batch_sen = np.array(eval_batch_sen)
            eval_batch_lan = np.array(eval_batch_lan)
            eval_batch_pos = np.array(eval_batch_pos)
            if (options.sentence_predict and epoch > options.non_neural_iter) or options.language_predict:
                batch_rule_samples = dmv_model.find_predict_samples(eval_batch_pos, eval_batch_lan, eval_batch_sen)
                batch_predict_data = utils.construct_ml_predict_data(batch_rule_samples)
                batch_predict_pos_v = torch.LongTensor(batch_predict_data['pos'])
                batch_predict_pos_index = np.array(batch_predict_data['pos'])
                batch_predict_dir_v = torch.LongTensor(batch_predict_data['dir'])
                batch_predict_dir_index = np.array(batch_predict_data['dir'])
                batch_predict_cvalency_v = torch.LongTensor(batch_predict_data['cvalency'])
                batch_predict_cvalency_index = np.array(batch_predict_data['cvalency'])
                batch_predict_lan_v = torch.LongTensor(batch_predict_data['languages'])
                batch_predict_lan_index = np.array(batch_predict_data['languages'])
                batch_predict_sen_v = []
                for sentence_id in batch_predict_data['sentence']:
                    batch_predict_sen_v.append(eval_sentence_map[sentence_id])
                batch_predict_sen_index = np.array(batch_predict_data['sentence'])
                batch_predict_sen_v = torch.LongTensor(batch_predict_sen_v)
                batch_predicted, batch_predicted_lan = m_model.forward_(batch_predict_pos_v, batch_predict_dir_v,
                                                                        batch_predict_cvalency_v, None, None, True,
                                                                        'child',
                                                                        batch_predict_lan_v, batch_predict_sen_v, None)
                if options.sentence_predict or options.language_predict:
                    # Evaluation of language pediction
                    for i in range(len(batch_predict_sen_v)):
                        sentence_idx = batch_predict_data['sentence'][i]
                        classify_results[sentence_idx] = batch_predicted_lan[i]
                    if options.sentence_predict:
                        eval_trans_param[batch_predict_sen_index, batch_predict_pos_index, :,
                        batch_predict_dir_index, batch_predict_cvalency_index] = batch_predicted.detach().numpy()
                else:
                    eval_trans_param[batch_predict_pos_index, :, batch_predict_dir_index, batch_predict_cvalency_index,
                    batch_predict_lan_index] = batch_predicted.detach().numpy()
            batch_score, batch_decision_score = dmv_model.evaluate_batch_score(eval_batch_pos, eval_batch_sen,
                                                                               eval_language_map, eval_languages,
                                                                               eval_trans_param)
            if options.function_mask:
                batch_score = dmv_model.function_to_mask(batch_score, eval_batch_pos)
            batch_score = np.expand_dims(batch_score, 3)
            batch_score = np.expand_dims(batch_score, 4)
            batch_decision_score = np.expand_dims(batch_decision_score, 2)
            batch_parse = eisner_for_dmv.batch_parse(batch_score, batch_decision_score, dmv_model.dvalency,
                                                     dmv_model.cvalency)
            for i in range(len(eval_batch_pos)):
                parse_results[eval_batch_sen[i]] = (batch_parse[0][i], batch_parse[1][i])
        utils.eval_ml(parse_results, eval_sentences, devpath, options.log + '_dev' + str(options.sample_idx),
                      eval_language_map, eval_languages, epoch)
        # utils.write_distribution(dmv_model)
        print "===================================="
        # language classification results
        if not options.eval_new_language and (options.sentence_predict or options.language_predict):
            correct = 0
            for i in range(len(classify_results)):
                if classify_results[i] == languages[eval_language_map[i]]:
                    correct += 1
            correct_rate = float(correct) / len(classify_results)
            print "Language classification accuracy " + str(correct_rate)


    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        print 'To use gpu' + str(options.gpu)

    chosen_list = ['en', 'de', 'nl', 'it', 'fr', 'la_ittb', 'no', 'bg', 'sl', 'grc',
                   'eu', 'et', 'fi', 'hi', 'ja']
    #chosen_list = ['en','nl']
    if not options.chosen_list:
        language_set = utils.read_language_list(options.language_path)
    else:
        language_set = set(chosen_list)

    file_list = os.listdir(options.train)
    file_set = utils.get_file_set(file_list, language_set, True)
    pos, sentences, languages, language_map = utils.read_multiple_data(options.train, file_set, False)
    sentence_language_map = {}
    if options.concat_all:
        languages = {'all': 0}
        for s in language_map.keys():
            language_map[s] = 'all'
    print 'Data read'
    with open(os.path.join(options.output, options.params + '_' + str(options.sample_idx)), 'w') as paramsfp:
        pickle.dump((pos, options), paramsfp)
    print 'Parameters saved'

    data_list, data_pos, sentence_map = utils.construct_ml_pos_data(sentences, pos, languages, language_map)
    batch_data = utils.construct_update_batch_data(data_list, options.batchsize)
    print 'Batch data constructed'
    data_size = len(data_list)

    ml_dmv_model = MLDMV(pos, sentence_map, languages, language_map, data_size, options)

    print 'Model constructed'

    ml_dmv_model.init_param(sentences)

    print 'Parameters initialized'
    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        ml_dmv_model.cuda(options.gpu)

    if options.use_neural:
        m_model = MMODEL(len(pos), len(languages), sentence_map, language_map, languages, options)
    else:
        m_model = None

    for epoch in range(options.epochs):
        print "\n"
        print "Training epoch " + str(epoch)
        ml_dmv_model.train()
        en_likehood = 0.0
        training_likelihood = 0.0
        trans_counter = np.zeros(
            (len(pos.keys()), len(pos.keys()), 2, options.c_valency, len(languages)))  # p c d v l
        # head_pos,head_tag,direction,decision_valence,decision,languages
        decision_counter = np.zeros((len(pos.keys()), 2, options.d_valency, 2, len(languages)))  # p d v stop l
        random.shuffle(batch_data)
        tot_batch = len(batch_data)
        if options.use_neural:
            ml_dmv_model.rule_samples = []
            ml_dmv_model.decision_samples = []

        for batch_id, one_batch in tqdm(enumerate(batch_data), mininterval=2,
                                        desc=' -Tot it %d (epoch %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
            batch_likelihood = 0.0
            sub_batch_data = utils.construct_batch_data(one_batch, options.sub_batch_size)
            # For each batch,put all sentences with the same length to one sub-batch
            for one_sub_batch in sub_batch_data:
                sub_batch_pos, sub_batch_lan, sub_batch_sen = [s[0] for s in one_sub_batch], \
                                                              [s[1] for s in one_sub_batch], \
                                                              [s[2][0] for s in one_sub_batch]
                # E-step
                sub_batch_likelihood, en_like = ml_dmv_model.em_e(sub_batch_pos, sub_batch_lan, sub_batch_sen,
                                                                  trans_counter, decision_counter, ml_dmv_model.em_type)
                en_likehood += en_like
                batch_likelihood += sub_batch_likelihood
            training_likelihood += batch_likelihood
        if epoch > options.non_neural_iter:
            ml_dmv_model.initial_flag = False
        print 'Likelihood for this iteration', training_likelihood
        # M-step
        # Using neural networks to update DMV parameters
        if options.use_neural:
            m_model.batch_training(ml_dmv_model.rule_samples, ml_dmv_model.decision_samples, data_pos)
        if options.use_neural and epoch > options.non_neural_iter:
            if options.sentence_predict:
                copy_trans_param = ml_dmv_model.sentence_trans_param.copy()
            else:
                copy_trans_param = ml_dmv_model.trans_param.copy()
            copy_decision_param = ml_dmv_model.decision_param.copy()
            # Predict model parameters by network
            trans_param, decision_param = m_model.predict(copy_trans_param, copy_decision_param,
                                                          decision_counter, options.child_only)
            if options.sentence_predict:
                ml_dmv_model.sentence_trans_param = trans_param.copy()
            else:
                ml_dmv_model.trans_param = trans_param.copy()
            ml_dmv_model.decision_param = decision_param.copy()
            print('PREDICT DONE ......')
        else:
            ml_dmv_model.em_m(trans_counter, decision_counter)  # TODO:
        if options.do_eval:
            do_eval(ml_dmv_model, m_model, pos, languages, language_map, epoch, options)
            # Save model parameters
    # with open(os.path.join(options.output, options.paramem) + str(epoch + 1) + '_' + str(options.sample_idx),
    #           'w') as paramem:
    #     pickle.dump(
    #         (ml_dmv_model.trans_param, ml_dmv_model.decision_param, ml_dmv_model.lex_param, ml_dmv_model.tag_num),
    #         paramem)
    ml_dmv_model.save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1) + '_' + str(
        options.sample_idx)))

print 'Training finished'
