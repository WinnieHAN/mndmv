import shutil

import numpy as np
import torch
import torch.nn as nn

import eisner_for_dmv
import utils
import m_dir


# from torch_model.NN_trainer import *


class ml_dmv_model(nn.Module):
    def __init__(self, pos, sentence_map, language_map, data_size, options):
        super(ml_dmv_model, self).__init__()
        self.options = options
        self.count_smoothing = options.count_smoothing
        self.param_smoothing = options.param_smoothing
        self.pos = pos
        self.sentence_map = sentence_map
        self.language_map = language_map
        # self.decision_pos = {}
        # self.to_decision = {}
        # self.from_decision = {}
        self.id_to_pos = {}
        self.em_type = options.em_type
        self.function_mask = options.function_mask
        self.use_neural = options.child_neural
        self.unified_network = options.unified_network
        self.sentence_score = {}
        self.sentence_counter = {}
        self.initial_flag = True
        self.data_size = data_size
        self.cvalency = options.c_valency
        self.dvalency = options.d_valency

        self.sentence_decision_counter = {}

        # self.child_neural = 1
        # self.root_neural = 0
        # self.decision_neural = 0

        self.trans_counter = np.zeros((len(pos.keys()), len(pos.keys()), 2, options.c_valency))  # p c d   # head_pos,head_tag,direction,decision_valence,decision
        self.root_counter = np.zeros(len(pos.keys()))
        self.decision_counter = np.zeros((len(pos.keys()), 2, options.d_valency, 2))  # p d v stop

        self.sentence_trans_param = np.zeros((data_size, len(pos.keys()), len(pos.keys()), 2, self.cvalency))  # p
        self.sentence_root_param = np.zeros((data_size, len(pos.keys())))
        self.sentence_decision_param = np.zeros((data_size, len(pos.keys()), 2, self.dvalency, 2))

        # p_counter = 0
        for p in pos.keys():
            self.id_to_pos[self.pos[p]] = p
            # p_id = self.pos[p]
            # self.decision_pos[p] = p_counter
            # self.to_decision[p_id] = p_counter  # tag id from child to decision
            # self.from_decision[p_counter] = p_id  # tag id from decision to child
            # p_counter += 1
        # self.dvalency = options.d_valency  # hanwj
        # self.cvalency = options.c_valency  # hanwj
        # head_pos,child_pos,direction,child_valence
        self.trans_param = np.zeros((len(pos), len(pos), 2, self.cvalency))
        self.root_param = np.zeros(len(pos))
        # head_pos,head_subtag,direction,valence,decision
        self.decision_param = np.zeros((len(pos), 2, self.dvalency, 2))

        self.trans_alpha = None
        self.use_prior = options.use_prior

        if self.function_mask:
            self.function_set = set()
            self.function_set.add("ADP")
            self.function_set.add("AUX")
            self.function_set.add("CONJ")
            self.function_set.add("DET")
            self.function_set.add("PART")
            self.function_set.add("SCONJ")

        if self.use_neural:
            self.rule_samples = list()
            self.decision_samples = list()

    def calcu_counter(self, best_parse, trans_counter, decision_counter, batch_pos):
        for sen_id in range(len(batch_pos)):
            pos_sentence = batch_pos[sen_id]
            heads = best_parse[0][sen_id]
            tags = best_parse[1][sen_id]
            head_valences = best_parse[2][sen_id]
            valences = best_parse[3][sen_id]
            for i, h in enumerate(heads):
                m_tag_id = int(tags[i])
                m_pos = pos_sentence[i]
                if h == -1:
                    continue
                m_dec_pos = self.to_decision[m_pos]
                m_head_valence = int(head_valences[i])
                m_valence = valences[i]
                if self.cvalency > 1:
                    m_child_valence = m_head_valence
                else:
                    m_child_valence = 0
                h = int(h)
                h_pos = pos_sentence[h]

                if h < i:
                    dir = 1
                else:
                    dir = 0
                h_tag_id = int(tags[h])
                trans_counter[h_pos, m_pos, h_tag_id, m_tag_id, dir, m_child_valence] += 1.
                decision_counter[m_dec_pos, m_tag_id, 0, int(m_valence[0]), 0] += 1.
                decision_counter[m_dec_pos, m_tag_id, 1, int(m_valence[1]), 0] += 1.
                if h > 0:
                    h_dec_pos = self.to_decision[h_pos]
                    decision_counter[h_dec_pos, h_tag_id, dir, m_head_valence, 1] += 1.

    def init_expert(self, data_f, pos):
        trans_counter = np.zeros(self.trans_param.shape)
        decision_counter = np.zeros(self.decision_param.shape)
        eval_sentences = utils.read_data(data_f, True)
        best_parse, batch_pos = [], []
        for s in eval_sentences:
            s_pos, s_parse = s.set_data_tag_parse_list(pos)
            batch_pos.append(s_pos)
            best_parse.append(s_parse)
        self.calcu_counter(best_parse, trans_counter, decision_counter, batch_pos)
        trans_sum = np.sum(self.trans_param, axis=1).reshape((len(self.pos), 1, 2, self.cvalency))
        decision_sum = np.sum(self.decision_param, axis=3).reshape((len(self.pos), 2, self.dvalency, 1))
        self.trans_param = self.trans_param / trans_sum
        self.decision_param = self.decision_param / decision_sum

    # KM initialization
    def init_param(self, data):
        # root_idx = self.pos['ROOT-POS']
        count_smoothing = 0.1
        norm_counter = np.zeros((len(self.pos), 2, self.dvalency, 2))  # pos_tag,direction,valence,decision
        for sentence in data:
            word_num = sentence.size
            change = np.zeros((word_num, 2))
            for i, entry in enumerate(sentence.entries):
                pos_id = self.pos[entry.pos]
                self.root_param[pos_id] += 1.0/word_num
            for j, m_entry in enumerate(sentence.entries):
                child_sum = 0
                for i in range(sentence.size):
                    if i == j:
                        continue
                    child_sum += 1. / abs(i - j)
                if child_sum > 0:
                    scale = float(word_num - 1) / word_num * (1. / child_sum)  # ??
                else:
                    scale = 0
                for i, h_entry in enumerate(sentence.entries):
                    if i == j:
                        continue
                    if j < i:
                        dir = 0
                    else:
                        dir = 1
                    span = abs(i - j)
                    h_pos = h_entry.pos
                    m_pos = m_entry.pos
                    h_pos_id = self.pos.get(h_pos)
                    m_pos_id = self.pos.get(m_pos)
                    self.trans_param[h_pos_id, m_pos_id, dir, :] += 1. / span * scale
                    change[i, dir] += 1. / span * scale
            self.update_decision(change, norm_counter, sentence.entries)
        self.root_param +=count_smoothing
        self.trans_param += count_smoothing
        self.decision_param += count_smoothing
        es = self.first_child_update(norm_counter)
        pr_first_kid = 0.9 * es
        norm_counter = norm_counter * pr_first_kid
        norm_counter = norm_counter.reshape((len(self.pos), 2, self.dvalency, 2))
        self.decision_param = self.decision_param + norm_counter
        # self.trans_param[:, root_idx, :, :] = 0
        # self.trans_param[root_idx, :, :, :, 0, :] = 0
        trans_sum = np.sum(self.trans_param, axis=1).reshape((len(self.pos), 1, 2, self.cvalency))
        root_sum = np.sum(self.root_param, axis=0)
        decision_sum = np.sum(self.decision_param, axis=3).reshape(
            (len(self.pos), 2, self.dvalency, 1))

        self.trans_param = self.trans_param / trans_sum
        self.root_param = self.root_param / root_sum
        self.decision_param = self.decision_param / decision_sum

    def update_decision(self, change, norm_counter, entries):
        word_num, _ = change.shape
        for i in range(word_num):
            pos_id = self.pos.get(entries[i].pos)
            for dir in range(2):
                if change[i, dir] > 0:
                    norm_counter[pos_id, dir, 0, 1] += 1
                    norm_counter[pos_id, dir, 1, 1] += -1
                    self.decision_param[pos_id, dir, 1, 1] += change[i, dir]

                    norm_counter[pos_id, dir, 0, 0] += -1
                    norm_counter[pos_id, dir, 1, 0] += 1
                    self.decision_param[pos_id, dir, 0, 0] += 1
                else:
                    self.decision_param[pos_id, dir, 0, 0] += 1

    def first_child_update(self, norm_counter):
        es = 1.0
        all_param = np.copy(self.decision_param)
        all_param = all_param.flatten()
        all_norm = norm_counter.flatten()
        for i in range(len(all_param)):
            if all_param[i] > 0:
                ratio = -all_param[i] / all_norm[i]
                if all_norm[i] < 0 and es > ratio:
                    es = ratio
        return es

    def em_e(self, batch_pos, batch_lan, batch_sen, em_type):
        batch_pos = np.array(batch_pos)
        if em_type == 'viterbi':
            # batch_likelihood = self.run_viterbi_estep(batch_pos, batch_lan, batch_sen, trans_counter,
            #                                           decision_counter)
            pass
        elif em_type == 'em':
            batch_likelihood = self.run_em_estep(batch_pos, batch_lan, batch_sen)

        return batch_likelihood

    def run_viterbi_estep(self, batch_pos, batch_words, batch_sen, trans_counter, decision_counter, lex_counter):
        batch_size = len(batch_pos)
        batch_score, batch_decision_score = self.evaluate_batch_score(batch_words, batch_pos)
        batch_score = np.array(batch_score)
        batch_decision_score = np.array(batch_decision_score)
        batch_score[:, :, 0, :, :, :] = -np.inf
        if self.specify_splitting:
            batch_score, batch_decision_score = self.mask_scores(batch_score, batch_decision_score, batch_pos)
        if self.function_mask:
            batch_score = self.function_to_mask(batch_score, batch_pos)

        best_parse = eisner_for_dmv.batch_parse(batch_score, batch_decision_score, self.dvalency, self.cvalency)

        batch_likelihood = self.update_counter(best_parse, trans_counter, decision_counter, lex_counter, batch_pos,
                                               batch_words)

        self.trans_counter = trans_counter
        return batch_likelihood

    def run_em_estep(self, batch_pos, batch_lan, batch_sen):
        batch_score, batch_root_score, batch_decision_score = self.evaluate_batch_score(batch_pos, batch_sen, self.sentence_trans_param)
        batch_score = np.array(batch_score)  # log p : batch_size, sentence_length, _, v_c_num
        batch_root_score = np.array(batch_root_score)
        batch_decision_score = np.array(batch_decision_score)
        if self.function_mask:
            batch_score = self.function_to_mask(batch_score, batch_pos)
        batch_size, sentence_length, _, v_c_num = batch_score.shape
        _, _, _, v_d_num, _ = batch_decision_score.shape
        # partial code is consistent with eisner_for_dmv.py (only child decision without root)
        batch_score = np.concatenate((np.full((batch_size, 1, sentence_length, v_c_num), -np.inf), batch_score),axis=1)# for eisner
        batch_score = np.concatenate((np.full((batch_size, sentence_length+1, 1, v_c_num), -np.inf), batch_score),axis=2)# for eisner
        batch_score[:, 0, 1:, 0] = batch_root_score
        batch_score[:, 0, 1:, 1] = batch_root_score
        batch_decision_score = np.concatenate((np.zeros((batch_size, 1, 2, v_d_num, 2)), batch_decision_score), axis=1)#np.concatenate((np.full((batch_size, 1, 2, v_d_num, 2), 0), batch_decision_score), axis=1)

        inside_batch_score = batch_score.reshape(batch_size, sentence_length+1, sentence_length+1, 1, 1, self.cvalency)
        inside_batch_decision_score = batch_decision_score.reshape(batch_size, sentence_length+1, 1, 2, self.dvalency, 2)
        inside_complete_table, inside_incomplete_table, sentence_prob = \
            eisner_for_dmv.batch_inside(inside_batch_score, inside_batch_decision_score, self.dvalency, self.cvalency)
        outside_complete_table, outside_incomplete_table = \
            eisner_for_dmv.batch_outside(inside_complete_table, inside_incomplete_table, inside_batch_score,
                                         inside_batch_decision_score, self.dvalency, self.cvalency)
        # update count and samples
        # self.rule_samples.append(list([h_pos, m_pos, dir, v, sentence_id, language_id, count]))
        # decision_counter
        # trans_counter
        # trans_counter_temp = np.zeros((self.trans_counter.shape[0]+1, self.trans_counter.shape[1]+1, self.trans_counter.shape[2], self.trans_counter.shape[3]))
        batch_pos_add_root = np.array([[self.trans_counter.shape[0]] + list(i) for i in batch_pos])
        batch_likelihood = self.update_pseudo_count(inside_incomplete_table, inside_complete_table, sentence_prob,
                                                    outside_incomplete_table, outside_complete_table, self.trans_counter, self.root_counter,
                                                    self.decision_counter, batch_pos_add_root, batch_sen, batch_lan)
        # self.root_counter += trans_counter_temp[0, 1:,1, 0]  # p c d v. direction of root is RIGHT, so dir is 1.
        # self.trans_counter += trans_counter_temp[1:, 1:, :, :]
        return batch_likelihood

    def evaluate_batch_score(self, batch_pos, batch_sen, sentence_trans_param):
        batch_size, sentence_length = batch_pos.shape
        # batch,head,child,head_tag,child_tag
        scores = np.zeros((batch_size, sentence_length, sentence_length, self.cvalency))
        root_scores = np.zeros((batch_size, sentence_length))
        # batch,position,tag,direction,valency,decision
        decision_scores = np.zeros((batch_size, sentence_length, 2, self.dvalency, 2))
        scores.fill(-np.inf)
        root_scores.fill(-np.inf)
        decision_scores.fill(-np.inf)
        for s in range(batch_size):
            sentence_id = batch_sen[s]
            for i in range(sentence_length):
                pos_id = batch_pos[s][i]
                decision_scores[s, i, :, :, :] = np.log(self.decision_param[pos_id, :, :, :])
                root_scores[s, i] = np.log(self.root_param[pos_id])
                for j in range(sentence_length):
                    h_pos_id = batch_pos[s][i]
                    m_pos_id = batch_pos[s][j]
                    if i == j:
                        continue
                    if i > j:
                        dir = 0
                    else:
                        dir = 1
                    if self.initial_flag:  #TODO True: #
                        scores[s, i, j, :] = np.log(self.trans_param[h_pos_id, m_pos_id, dir, :])
                    else:
                        scores[s, i, j, :] = np.log(sentence_trans_param[sentence_id, h_pos_id, m_pos_id, dir, :])
        return scores, root_scores, decision_scores  # scores: batch, h, c, v  ;decision_scores: batch, h d v stop

    def update_counter(self, best_parse, trans_counter, decision_counter, lex_counter, batch_pos, batch_words):
        batch_likelihood = 0.0
        for sen_id in range(len(batch_pos)):
            pos_sentence = batch_pos[sen_id]
            word_sentence = batch_words[sen_id]
            heads = best_parse[0][sen_id]
            tags = best_parse[1][sen_id]
            head_valences = best_parse[2][sen_id]
            valences = best_parse[3][sen_id]
            for i, h in enumerate(heads):
                m_tag_id = int(tags[i])
                m_pos = pos_sentence[i]
                if h == -1:
                    continue
                m_dec_pos = self.to_decision[m_pos]
                m_head_valence = int(head_valences[i])
                m_valence = valences[i]
                if self.cvalency > 1:
                    m_child_valence = m_head_valence
                else:
                    m_child_valence = 0
                m_word = word_sentence[i]
                h = int(h)
                h_pos = pos_sentence[h]

                if h < i:
                    dir = 1
                else:
                    dir = 0
                h_tag_id = int(tags[h])
                trans_counter[h_pos, m_pos, h_tag_id, m_tag_id, dir, m_child_valence] += 1.
                batch_likelihood += np.log(self.trans_param[h_pos, m_pos, h_tag_id, m_tag_id, dir, m_child_valence])
                if self.use_neural:
                    self.rule_samples.append(list([h_pos, m_pos, h_tag_id, m_tag_id, dir, m_child_valence]))
                decision_counter[m_dec_pos, m_tag_id, 0, int(m_valence[0]), 0] += 1.
                decision_counter[m_dec_pos, m_tag_id, 1, int(m_valence[1]), 0] += 1.
                if self.use_neural:
                    if not self.unified_network:
                        self.decision_samples.append(list([m_dec_pos, m_tag_id, 0, int(m_valence[0]), 0]))
                        self.decision_samples.append(list([m_dec_pos, m_tag_id, 1, int(m_valence[1]), 0]))
                    else:
                        self.decision_samples.append(list([m_pos, m_tag_id, 0, int(m_valence[0]), 0]))
                        self.decision_samples.append(list([m_pos, m_tag_id, 1, int(m_valence[1]), 0]))
                # lexicon count
                if (self.use_lex):
                    lex_counter[m_pos, m_tag_id, m_word] += 1
                    batch_likelihood += np.log(self.lex_param[m_pos, m_tag_id, m_word])
                if h > 0:
                    h_dec_pos = self.to_decision[h_pos]
                    decision_counter[h_dec_pos, h_tag_id, dir, m_head_valence, 1] += 1.
                    if self.use_neural:
                        if not self.unified_network:
                            self.decision_samples.append(list([h_dec_pos, h_tag_id, dir, m_head_valence, 1]))
                        else:
                            self.decision_samples.append(list([h_pos, h_tag_id, dir, m_head_valence, 1]))
                    batch_likelihood += np.log(self.decision_param[h_dec_pos, h_tag_id, dir, m_head_valence, 1])
                batch_likelihood += np.log(self.decision_param[m_dec_pos, m_tag_id, 0, int(m_valence[0]), 0])
                batch_likelihood += np.log(self.decision_param[m_dec_pos, m_tag_id, 1, int(m_valence[1]), 0])

        return batch_likelihood

    def em_m(self, trans_counter, root_counter, decision_counter, lex_counter, child_model, decision_model):
        # if self.tag_num > 1:
        #     self.param_smoothing = 1e-8
        trans_counter = trans_counter + self.param_smoothing
        root_counter = root_counter + self.param_smoothing
        decision_counter = decision_counter + self.param_smoothing
        # if self.use_lex:
        #     lex_counter = lex_counter + self.param_smoothing
        child_sum = np.sum(trans_counter, axis=1, keepdims=True)#np.sum(trans_counter, axis=(1, 3)).reshape(len(self.pos), 1, self.tag_num, 1, 2, self.cvalency)
        decision_sum = np.sum(decision_counter, axis=3, keepdims=True)#np.sum(decision_counter, axis=4).reshape(len(self.pos), self.tag_num, 2, self.dvalency, 1)
        root_sum = np.sum(root_counter)
        self.trans_param = trans_counter / child_sum
        self.decision_param = decision_counter / decision_sum
        self.root_param = root_counter / root_sum
        # if self.use_lex:
        #     pass
            # lex_sum = np.sum(lex_counter, axis=2).reshape(len(self.pos), self.tag_num, 1)
            # self.lex_param = lex_counter / lex_sum
        return

    def update_pseudo_count(self, inside_incomplete_table, inside_complete_table, sentence_prob,
                            outside_incomplete_table, outside_complete_table, trans_counter, root_counter,
                            decision_counter, batch_pos, batch_sen, batch_lan):
        batch_likelihood = 0.0
        batch_size, sentence_length = batch_pos.shape
        span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(sentence_length, False)
        for s in range(batch_size):
            pos_sentence = batch_pos[s]
            sentence_id = batch_sen[s]
            language_id = batch_lan[s]
            one_sentence_count = []
            one_sentence_decision_count = []
            for h in range(sentence_length):
                for m in range(sentence_length):
                    if m == 0:
                        continue
                    if h == m:
                        continue
                    if h > m:
                        dir = 0
                    else:
                        dir = 1
                    h_pos = pos_sentence[h]

                    m_pos = pos_sentence[m]
                    m_dec_pos = m_pos
                    if dir == 0:
                        span_id = span_2_id[(m, h, dir)]
                    else:
                        span_id = span_2_id[(h, m, dir)]
                    dep_count = inside_incomplete_table[s, span_id, :, :, :] + \
                                outside_incomplete_table[s, span_id, :, :, :] - sentence_prob[s]
                    if dir == 0:
                        dep_count = dep_count.swapaxes(1, 0)
                    if self.cvalency == 1:
                        if h == 0:
                            root_counter[m_pos] += np.sum(np.exp(dep_count))
                        else:
                            trans_counter[h_pos, m_pos, dir, 0] += np.sum(np.exp(dep_count))
                    else:
                        if h == 0:
                            root_counter[m_pos] += np.sum(np.exp(dep_count))
                        else:
                            trans_counter[h_pos, m_pos, dir, :] += np.exp(dep_count).reshape(self.cvalency)
                    if self.use_neural:
                        for v in range(self.cvalency):
                            count = np.exp(dep_count).reshape(self.cvalency)[v]
                            if not h == 0:
                                self.rule_samples.append(list([h_pos, m_pos, dir, v, sentence_id, language_id, count]))
                    if h > 0:
                        h_dec_pos = h_pos
                        decision_counter[h_dec_pos, dir, :, 1] += np.exp(dep_count).reshape(self.cvalency)
                        if self.use_neural:
                            reshaped_count = np.exp(dep_count).reshape(self.cvalency)
                            for v in range(self.dvalency):
                                count = reshaped_count[v]
                                if not self.unified_network:
                                    self.decision_samples.append(
                                        list([h_dec_pos, dir, v, sentence_id, language_id, 1, count]))
                                else:
                                    self.decision_samples.append(
                                        list([h_pos, dir, v, sentence_id, language_id, 1, count]))
            for m in range(1, sentence_length):
                m_pos = pos_sentence[m]
                m_dec_pos = m_pos
                for d in range(2):
                    m_span_id = span_2_id[(m, m, d)]
                    stop_count = inside_complete_table[s, m_span_id, :, :] + \
                                 outside_complete_table[s, m_span_id, :, :] - sentence_prob[s]
                    decision_counter[m_dec_pos, d, :, 0] += np.exp(stop_count).reshape(self.cvalency)
                    if self.use_neural:
                        for v in range(self.dvalency):
                            count = np.exp(stop_count).reshape(self.cvalency)[v]
                            if not self.unified_network:
                                self.decision_samples.append(
                                    list([m_dec_pos, d, v, sentence_id, language_id, 0, count]))
                            else:
                                self.decision_samples.append(list([m_pos, d, v, 0, sentence_id, language_id, count]))

            batch_likelihood += sentence_prob[s]
            self.sentence_counter[sentence_id] = one_sentence_count
            self.sentence_decision_counter[sentence_id] = one_sentence_decision_count
        return batch_likelihood

    def find_predict_samples(self, batch_pos, batch_sen):
        batch_size, sentence_length = batch_pos.shape
        predict_rule_samples = []
        rule_involved = set()
        for s in range(batch_size):
            pos_sentence = batch_pos[s]
            sentence_id = batch_sen[s]
            for h in range(sentence_length):
                for dir in range(2):
                    if h == 0 and dir == 0:  # TODO for what ?
                        continue
                    h_pos = pos_sentence[h]
                    for v in range(self.cvalency):
                        rule_tuple = (h_pos, dir, v, sentence_id)
                        if rule_tuple not in rule_involved:
                            rule_involved.add(rule_tuple)
                            predict_rule_samples.append(list([h_pos, dir, v, sentence_id]))
        return np.array(predict_rule_samples)

    def mask_scores(self, batch_scores, batch_decision_scores, batch_pos):
        batch_size, sentence_length, _, _, _, _ = batch_scores.shape
        score_mask = np.zeros((batch_size, sentence_length, sentence_length, self.tag_num, self.tag_num, self.cvalency))
        decision_mask = np.zeros((batch_size, sentence_length, self.tag_num, 2, self.dvalency, 2))
        for s in range(batch_size):
            for i in range(sentence_length):
                if self.id_to_pos[batch_pos[s, i]] not in self.specify_list:
                    decision_mask[s, i, 1:, :] = -np.inf
                for j in range(sentence_length):
                    if self.id_to_pos[batch_pos[s, i]] not in self.specify_list and self.id_to_pos[batch_pos[s, j]] not in self.specify_list:
                        score_mask[s, i, j, :, :, :] = -np.inf
                        score_mask[s, i, j, 0, 0, :] = 0
                    if self.id_to_pos[batch_pos[s, i]] not in self.specify_list and self.id_to_pos[
                        batch_pos[s, j]] in self.specify_list:
                        score_mask[s, i, j, 1:, :, :] = -np.inf
                    if self.id_to_pos[batch_pos[s, i]] in self.specify_list and self.id_to_pos[
                        batch_pos[s, j]] not in self.specify_list:
                        score_mask[s, i, j, :, 1:, :] = -np.inf
        batch_scores = batch_scores + score_mask
        batch_decision_scores = batch_decision_scores + decision_mask
        return batch_scores, batch_decision_scores

    def function_to_mask(self, batch_score, batch_pos):
        batch_size, sentence_length, _, _ = batch_score.shape
        for s in range(batch_size):
            for i in range(sentence_length):
                pos_id = batch_pos[s, i]
                pos = self.id_to_pos[pos_id]
                if pos in self.function_set:
                    batch_score[s, i, :, :] = -1e5#-np.inf
        return batch_score

    def apply_prior(self, trans_counter, lex_counter, prior_alpha, prior_epsilon, lex_prior_alpha, lex_epsilon):
        if self.trans_alpha is None:
            child_mean = np.average(trans_counter, axis=(1, 3)).reshape(len(self.pos), 1, self.tag_num, 1, 2,
                                                                        self.cvalency)
            self.trans_alpha = -child_mean * prior_alpha
        if self.tag_num > 1:
            prior_epsilon = 1e-3
        for h in range(len(self.pos)):
            for t in range(self.tag_num):
                for dir in range(2):
                    for c in range(self.cvalency):
                        dir_trans_alpha = trans_counter[h, :, t, :, dir, c] + self.trans_alpha[h, :, t, :, dir, c]
                        dim = self.tag_num * len(self.pos)
                        dir_trans_alpha = dir_trans_alpha.reshape(dim, 1)
                        md = m_dir.modified_dir(dim, dir_trans_alpha, prior_epsilon)
                        posterior_counts = md.get_mode()
                        trans_counter[h, :, t, :, dir, c] = posterior_counts.reshape(len(self.pos), self.tag_num)

        if self.use_lex and self.tag_num > 1:
            if self.lex_alpha is None:
                lex_mean = np.average(lex_counter, axis=2).reshape(len(self.pos), self.tag_num, 1)
                self.lex_alpha = -lex_mean * lex_prior_alpha
            for p in range(len(self.pos)):
                for t in range(self.tag_num):
                    dir_lex_alpha = lex_counter[p, t, :] + self.lex_alpha[p, t, :]
                    dim = len(self.vocab)
                    md = m_dir.modified_dir(dim, dir_lex_alpha, lex_epsilon)
                    posterior_counts = md.get_mode()
                    lex_counter[p, t, :] = posterior_counts

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))
