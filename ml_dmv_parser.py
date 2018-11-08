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
    # torch.manual_seed(1)
    parser = OptionParser()
    parser.add_option("--train", dest="train", help="train file", metavar="FILE", default="en") # data/ud_file
    parser.add_option("--dev", dest="dev", help="dev file", metavar="FILE",
                      default="en") # data/ud40_test

    parser.add_option("--batch", type="int", dest="batchsize", default=1000)
    parser.add_option("--sample_batch", type="int", dest="sample_batch_size", default=10000)

    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="output/dmv.model")

    parser.add_option("--pembedding", type="int", dest="pembedding_dim", default=10)
    parser.add_option("--epochs", type="int", dest="epochs", default=50)
    parser.add_option("--non_neural_iter", type="int", dest="non_neural_iter", default=20)  # if non_neural_iter>epochs, then it is DMV.
    parser.add_option("--non_dscrm_iter", type="int", dest="non_dscrm_iter", default=40)  # if non_neural_iter>epochs, then it is DMV.
    parser.add_option("--tag_num", type="int", dest="tag_num", default=1)

    parser.add_option("--dvalency", type="int", dest="d_valency", default=2)
    parser.add_option("--cvalency", type="int", dest="c_valency", default=1)
    parser.add_option("--em_type", type="string", dest="em_type", default='viterbi')

    parser.add_option("--count_smoothing", type="float", dest="count_smoothing", default=0.1)
    parser.add_option("--param_smoothing", type="float", dest="param_smoothing", default=1e-8)

    parser.add_option("--optim", type="string", dest="optim_type", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.001)
    parser.add_option("--outdir", type="string", dest="output", default="output")

    parser.add_option("--sample_idx", type="int", dest="sample_idx", default=1000)
    parser.add_option("--prior_alpha", type="float", dest="prior_alpha", default=0.0)
    parser.add_option("--do_eval", action="store_true", dest="do_eval", default=False)
    parser.add_option("--log", dest="log", help="log file", metavar="FILE", default="output/log")
    parser.add_option("--sub_batch", type="int", dest="sub_batch_size", default=100)
    parser.add_option("--use_prior", action="store_true", dest="use_prior", default=False)
    parser.add_option("--prior_epsilon", type="float", dest="prior_epsilon", default=1)

    parser.add_option("--function_mask", action="store_true", default=False)

    parser.add_option("--e_pass", type="int", dest="e_pass", default=4)
    parser.add_option("--em_iter", type="int", dest="em_iter", default=1)

    parser.add_option("--paramem", dest="paramem", help="EM parameters file", metavar="FILE",
                      default="paramem.pickle")

    parser.add_option("--gpu", type="int", dest="gpu", default=-1, help='gpu id, set to -1 if use cpu mode')

    parser.add_option("--seed", type="int", dest="seed", default=0)
    parser.add_option("--drop_out", type="float", dest="drop_out", default=0.25)

    parser.add_option("--child_only", action="store_true", dest="child_only", default=False)  # TODO True
    parser.add_option("--valency_dim", type="int", dest="valency_dim", default=5)
    parser.add_option("--hid_dim", type="int", dest="hid_dim", default=10)
    parser.add_option("--pre_ouput_dim", type="int", dest="pre_output_dim", default=15)
    parser.add_option("--decision_pre_output_dim", type="int", dest="decision_pre_output_dim", default=5)
    parser.add_option("--neural_epoch", type="int", dest="neural_epoch", default=1)
    parser.add_option("--unified_network", action="store_true", dest="unified_network", default=False)
    parser.add_option("--reset_weight", action="store_true", dest="reset_weight", default=False)
    parser.add_option("--dir_embed", action="store_true", dest="dir_embed", default=False)
    parser.add_option("--dir_dim", type="int", dest="dir_dim", default=1)
    # parser.add_option("--use_neural", action="store_true", dest="use_neural", default=False)

    parser.add_option("--child_neural", action="store_true", dest="child_neural", default=False)
    parser.add_option("--root_neural", action="store_true", dest="root_neural", default=False)
    parser.add_option("--decision_neural", action="store_true", dest="decision_neural", default=False)

    parser.add_option("--language_path", type="string", dest="language_path", default="data/ud-treebanks-v1.4") # data/language_list
    parser.add_option("--ml_comb_type", type="int", dest="ml_comb_type", default=1) # options.ml_comb_type = 0(no_lang_id)/1(id embeddings)/2(classify-tags
    parser.add_option("--stc_model_type", type="int", dest="stc_model_type", default=1) # 1  lstm   2 lstm with atten   3 variational
    parser.add_option("--lang_dim", type="int", dest="lang_dim", default=5)
    parser.add_option("--lstm_layer_num", type="int", dest="lstm_layer_num", default=1)
    parser.add_option("--lstm_hidden_dim", type="int", dest="lstm_hidden_dim", default=10)
    parser.add_option("--bidirectional", action="store_true", dest="bidirectional", default=False)

    parser.add_option("--load_model", action="store_true", dest="load_model", default=False)
    parser.add_option("--loaded_model_idx", type="int", dest="loaded_model_idx", default=1000)

    (options, args) = parser.parse_args()

    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        print 'To use gpu' + str(options.gpu)
    else:
        torch.set_num_threads(5)


    def do_eval(dmv_model, m_model, pos, options, epoch):
        print "===================================="
        print 'Do evaluation on development set'
        # eval_sentences = utils.read_data(options.dev, True)
        if not options.load_model:
            ml_sentences = utils.read_ml_corpus(options.language_path, options.dev, stc_length=40, isPredict=True, isadd=False)
        else:
            ml_sentences = utils.read_ml_corpus(options.language_path, options.dev, stc_length=40, isPredict=True, isadd=True)
        eval_sentences = ml_sentences[0]
        dmv_model.eval()
        eval_sentence_map = {}
        eval_sen_idx = 0
        eval_data_list = list()
        devpath = os.path.join(options.output, 'eval_pred' + str(epoch + 1) + '_' + str(options.sample_idx))
        lang_id = languages[options.dev] if options.dev in languages else 0  # 0 is manually specified (when dev_lang is not trained before)
        for s in eval_sentences:
            _, s_pos = s.set_data_list(None, pos)
            s_data_list = list()
            s_data_list.append(s_pos)
            s_data_list.append([eval_sen_idx])
            eval_data_list.append(s_data_list)
            eval_sentence_map[eval_sen_idx] = s_pos
            eval_sen_idx += 1
        eval_batch_data = utils.construct_batch_data(eval_data_list, options.batchsize)
        parse_results = {}
        eval_sentence_trans_param = np.zeros(
            (len(eval_data_list), len(pos.keys()), len(pos.keys()), 2, options.c_valency))
        for batch_id, one_batch in enumerate(eval_batch_data):
            eval_batch_pos, eval_batch_sen = [s[0] for s in one_batch], [s[1][0] for s in one_batch]
            eval_batch_sen = np.array(eval_batch_sen)
            eval_batch_pos = np.array(eval_batch_pos)
            if dmv_model.initial_flag:
                batch_score, batch_root_score, batch_decision_score = dmv_model.evaluate_batch_score(eval_batch_pos, eval_batch_sen, None, epoch)
            else:
                batch_rule_samples = dmv_model.find_predict_samples(eval_batch_pos, eval_batch_sen)
                batch_predict_data = utils.construct_ml_predict_data(batch_rule_samples)
                batch_predict_pos_v = torch.LongTensor(batch_predict_data['pos'])
                batch_predict_pos_index = np.array(batch_predict_data['pos'])
                batch_predict_dir_v = torch.LongTensor(batch_predict_data['dir'])
                batch_predict_dir_index = np.array(batch_predict_data['dir'])
                batch_predict_cvalency_v = torch.LongTensor(batch_predict_data['cvalency'])
                batch_predict_cvalency_index = np.array(batch_predict_data['cvalency'])
                batch_predict_sen_v = []
                for sentence_id in batch_predict_data['sentence']:
                    batch_predict_sen_v.append(eval_sentence_map[sentence_id])
                batch_predict_sen_index = np.array(batch_predict_data['sentence'])
                batch_predict_sen_v = torch.LongTensor(batch_predict_sen_v)
                batch_predict_sen_len = torch.LongTensor(np.array([len(i) for i in batch_predict_sen_v]))
                batch_predict_lan_v = torch.LongTensor(np.array([lang_id for _ in batch_predict_sen_v]))  # TODO
                batch_predicted = m_model.forward_(batch_predict_pos_v, batch_predict_dir_v, batch_predict_cvalency_v,
                                                   None, None, True, 'child', options.em_type, batch_predict_lan_v, batch_predict_sen_v,
                                                   batch_predict_sen_len, epoch=epoch)

                eval_sentence_trans_param[batch_predict_sen_index, batch_predict_pos_index, :, batch_predict_dir_index,
                batch_predict_cvalency_index] = batch_predicted.detach().numpy()
                batch_score, batch_root_score, batch_decision_score = dmv_model.evaluate_batch_score(eval_batch_pos, eval_batch_sen, eval_sentence_trans_param, epoch)
            batch_size, sentence_length, _, v_c_num = batch_score.shape
            _, _, _, v_d_num, _ = batch_decision_score.shape

            batch_score = np.concatenate((np.full((batch_size, 1, sentence_length, v_c_num), -np.inf), batch_score),
                                         axis=1)  # for eisner
            batch_score = np.concatenate((np.full((batch_size, sentence_length + 1, 1, v_c_num), -np.inf), batch_score),
                                         axis=2)  # for eisner
            batch_score[:, 0, 1:, 0] = batch_root_score
            batch_decision_score = np.concatenate((np.zeros((batch_size, 1, 2, v_d_num, 2)), batch_decision_score),
                                                  axis=1)

            batch_score = np.expand_dims(batch_score,3)
            batch_score = np.expand_dims(batch_score,4)
            batch_decision_score = np.expand_dims(batch_decision_score,2)
            batch_parse = eisner_for_dmv.batch_parse(batch_score, batch_decision_score, dmv_model.dvalency,
                                                     dmv_model.cvalency)
            for i in range(len(eval_batch_pos)):
                parse_results[eval_batch_sen[i]] = (batch_parse[0][i], batch_parse[1][i])
        utils.eval(parse_results, eval_sentences, devpath, options.log + '_dev' + str(options.sample_idx), epoch)
        # utils.write_distribution(dmv_model)
        print "===================================="

    if not options.load_model:
        pos, sentences, languages, language_map = utils.read_ml_corpus(options.language_path, options.train, stc_length=15, isPredict=False, isadd=False) # pos: str 2 id, sentences, languages: lang 2 id, language_map: stc_id 2 lang
    else:
        pos, sentences, languages, language_map = utils.read_ml_corpus(options.language_path, options.train, stc_length=15, isPredict=False, isadd=True)  # language: lang2i

    sentence_language_map = {}
    # print 'Data read'
    # with open(os.path.join(options.output, options.params + '_' + str(options.sample_idx)), 'w') as paramsfp:
    #     pickle.dump((pos, options), paramsfp)
    # print 'Parameters saved'

    data_list = list()
    sen_idx = 0
    sentence_map = {} # id 2 pos_seq
    data_pos = []
    for s in sentences:
        _, s_pos = s.set_data_list(None, pos)
        s_data_list = list()
        s_data_list.append(s_pos)
        data_pos.append(s_pos)
        s_data_list.append(languages[language_map[sen_idx]])
        s_data_list.append([sen_idx])
        data_list.append(s_data_list)
        sentence_map[sen_idx] = s_pos
        sen_idx += 1
    data_pos = np.array(data_pos)  # list of sentences with only tags
    batch_data = utils.construct_update_batch_data(data_list, options.batchsize)  # data_list: tag_seq, lang_id, stc_id
    print 'Batch data constructed'
    data_size = len(data_list)

    print 'Model constructed'
    load_file = os.path.join(options.output, options.paramem) + '_' + str(options.loaded_model_idx)
    ml_dmv_model = MLDMV(pos, sentence_map, language_map, data_size, options)
    if (not options.load_model) or (not os.path.exists(load_file)):
        ml_dmv_model.init_param(sentences)
    else:
        ml_dmv_model.trans_param, ml_dmv_model.root_param, ml_dmv_model.decision_param, ml_dmv_model.sentence_trans_param = pickle.load(open(load_file, 'r'))
    epoch = -1
    do_eval(ml_dmv_model, None, pos, options, epoch)

    print 'Decoder parameters initialized'
    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        ml_dmv_model.cuda(options.gpu)

    loaded_file = os.path.join(options.output, os.path.basename(options.model) + '_' + str(options.loaded_model_idx))
    if (not options.load_model) or (not os.path.exists(loaded_file)):
        # if options.child_neural or options.decision_neural:
        m_model = MMODEL(len(pos), len(languages), options)
    else:
        m_model = torch.load(loaded_file)


    for epoch in range(options.epochs):
        print "\n"
        print "Training epoch " + str(epoch)
        ml_dmv_model.train()

        for n in range(options.em_iter):
            print 'em iteration ', n
            training_likelihood = 0.0

            # trans_counter = np.zeros(
            #     (len(pos.keys()), len(pos.keys()), 2, options.c_valency))  # p c d v
            # # head_pos,head_tag,direction,decision_valence,decision
            # root_counter = np.zeros(len(pos.keys()))
            # decision_counter = np.zeros((len(pos.keys()), 2, options.d_valency, 2))  # p d v stop

            random.shuffle(batch_data) #TODO hanwj
            tot_batch = len(batch_data)
            ml_dmv_model.root_counter.fill(0)
            ml_dmv_model.trans_counter.fill(0)
            ml_dmv_model.decision_counter.fill(0)
            ml_dmv_model.rule_samples = []
            ml_dmv_model.decision_samples = []
            for batch_id, one_batch in tqdm(
                    enumerate(batch_data), mininterval=2,
                    desc=' -Tot it %d (epoch %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
                batch_likelihood = 0.0
                sub_batch_data = utils.construct_batch_data(one_batch, options.sub_batch_size)
                # For each batch,put all sentences with the same length to one sub-batch
                for one_sub_batch in sub_batch_data:
                    sub_batch_pos, sub_batch_lan, sub_batch_sen = [s[0] for s in one_sub_batch], \
                                                                  [s[1] for s in one_sub_batch], \
                                                                  [s[2][0] for s in one_sub_batch]
                    # E-step
                    sub_batch_likelihood = ml_dmv_model.em_e(sub_batch_pos, sub_batch_lan, sub_batch_sen, ml_dmv_model.em_type, epoch)
                    batch_likelihood += sub_batch_likelihood
                training_likelihood += batch_likelihood
                # print(batch_id)
                # print(batch_likelihood)
            ml_dmv_model.initial_flag = False
            print 'Likelihood for this iteration', training_likelihood
            # M-step
            # Using neural networks to update DMV parameters
            if options.child_neural or options.decision_neural:
                for e in range(options.neural_epoch):

                    iter_loss = 0.0
                    # Put training samples in batches
                    batch_input_data, batch_target_data, batch_decision_data, batch_decision_target_data = \
                        utils.construct_ml_input_data(ml_dmv_model.rule_samples, ml_dmv_model.decision_samples,
                                                      sentence_map, options.sample_batch_size, options.em_type)
                    print 'batch_data for M-step constructed'
                    batch_num = len(batch_input_data['input_pos'])
                    tot_batch = batch_num
                    for batch_id in tqdm(range(batch_num), mininterval=2, desc=' -Tot it %d (iter %d)' % (tot_batch, 0),
                                         leave=False, file=sys.stdout):
                        # Input for the network: head_pos,direction,child valency
                        batch_input_pos_v = torch.LongTensor(batch_input_data['input_pos'][batch_id])
                        batch_input_dir_v = torch.LongTensor(batch_input_data['input_dir'][batch_id])
                        batch_cvalency_v = torch.LongTensor(batch_input_data['cvalency'][batch_id])
                        batch_input_sen_v = []
                        for sentence_id in batch_input_data['sentence'][batch_id]:
                            batch_input_sen_v.append(data_pos[int(sentence_id)])
                        batch_input_sen_v = torch.LongTensor(batch_input_sen_v)
                        batch_target_lan_v = torch.LongTensor(batch_target_data['target_lan'][batch_id])
                        batch_target_pos_v = torch.LongTensor(batch_target_data['target_pos'][batch_id])
                        if options.em_type == 'em':
                            batch_target_pos_count_v = torch.FloatTensor(batch_target_data['target_count'][batch_id])
                        else:
                            batch_target_pos_count_v = None
                        batch_input_len = len(batch_input_sen_v[0])
                        batch_loss = m_model.forward_(batch_input_pos_v, batch_input_dir_v, batch_cvalency_v,
                                                      batch_target_pos_v, batch_target_pos_count_v, False, 'child',
                                                      options.em_type, batch_target_lan_v, batch_input_sen_v,
                                                      batch_input_len, epoch=epoch)
                        iter_loss += batch_loss
                        batch_loss.backward()
                        m_model.optim.step()
                        m_model.optim.zero_grad()
                    print "child loss for this iteration is " + str(iter_loss.detach().data.numpy() / batch_num)
                    # Network for decision distribution
                    if not options.child_only:
                        pass
                        # batch_num = len(batch_decision_data['decision_pos'])
                        # tot_batch = batch_num
                        # iter_decision_loss = 0.0
                        # for decision_batch_id in tqdm(
                        #         range(batch_num), mininterval=2,
                        #         desc=' -Tot it %d (iter %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
                        #     # Input for decision network: pos,direction,decision_valency
                        #     batch_decision_pos_v = torch.LongTensor(batch_decision_data['decision_pos'][batch_id])
                        #     batch_dvalency_v = torch.LongTensor(batch_decision_data['dvalency'][batch_id])
                        #     batch_decision_dir_v = torch.LongTensor(batch_decision_data['decision_dir'][batch_id])
                        #     batch_target_decision_v = torch.LongTensor(
                        #         batch_target_decision_data['decision_target'][batch_id])
                        #     if options.em_type == 'em':
                        #         batch_target_decision_count_v = torch.FloatTensor(
                        #             batch_target_decision_data['decision_target_count'][batch_id])
                        #     else:
                        #         batch_target_decision_count_v = None
                        #     if options.unified_network:
                        #         batch_decision_loss = m_model.forward_(batch_decision_pos_v, batch_decision_dir_v,
                        #                                                batch_dvalency_v, batch_target_decision_v,
                        #                                                batch_target_decision_count_v, False, 'decision',
                        #                                                options.em_type)
                        #     else:
                        #         batch_decision_loss = m_model.forward_decision(batch_decision_pos_v,
                        #                                                        batch_decision_dir_v, batch_dvalency_v,
                        #                                                        batch_target_decision_v,
                        #                                                        batch_target_decision_count_v, False,
                        #                                                        options.em_type)
                        #     iter_decision_loss += batch_decision_loss
                        #     batch_decision_loss.backward()
                        #     m_model.optim.step()
                        #     m_model.optim.zero_grad()
                        # print "decision loss for this iteration is " + str(
                        #     iter_decision_loss.detach().data.numpy() / batch_num)
                copy_sentence_trans_param = ml_dmv_model.sentence_trans_param.copy()
                copy_decision_param = ml_dmv_model.decision_param.copy()
                # copy_trans_param = ml_dmv_model.trans_param.copy()
                copy_root_param = ml_dmv_model.root_param.copy()
                # Predict model parameters by network
                sentence_trans_param, trans_param, root_param, decision_param = m_model.predict(copy_sentence_trans_param, copy_root_param, copy_decision_param,
                                                                                   options.sample_batch_size, ml_dmv_model.trans_counter, ml_dmv_model.root_counter, ml_dmv_model.decision_counter,
                                                                                   sentence_map, language_map, languages, epoch=epoch)
                ml_dmv_model.sentence_trans_param = sentence_trans_param.copy()
                ml_dmv_model.trans_param = trans_param.copy()
                ml_dmv_model.decision_param = decision_param.copy()
                ml_dmv_model.root_param = root_param.copy()
                print('PREDICT DONE ......')
            else:
                ml_dmv_model.em_m(ml_dmv_model.trans_counter, ml_dmv_model.root_counter, ml_dmv_model.decision_counter, None, None, None)  # TODO:
        if options.do_eval:
            do_eval(ml_dmv_model, m_model, pos, options, epoch)
            # Save model parameters
        with open(os.path.join(options.output, options.paramem) + '_' + str(options.sample_idx), 'w') as paramem:
            pickle.dump((ml_dmv_model.trans_param, ml_dmv_model.root_param, ml_dmv_model.decision_param, ml_dmv_model.sentence_trans_param), paramem)
        # ml_dmv_model.save(os.path.join(options.output, os.path.basename(options.model) + '_' + str(options.sample_idx)))
        torch.save(m_model, os.path.join(options.output, os.path.basename(options.model) + '_' + str(options.sample_idx)))

    print 'Training finished'
