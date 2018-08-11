import shutil

import numpy as np
import torch
import torch.nn as nn

import eisner_for_dmv
import utils
import m_dir
from torch_model.NN_trainer import *


class ldmv_model(nn.Module):
    def __init__(self, vocab, pos, options):
        super(ldmv_model, self).__init__()
        self.options = options
        self.tag_num = options.tag_num
        self.count_smoothing = options.count_smoothing
        self.param_smoothing = options.param_smoothing
        self.vocab = vocab
        self.pos = pos
        self.decision_pos = {}
        self.to_decision = {}
        self.from_decision = {}
        self.id_to_pos = {}
        self.use_lex = options.use_lex
        self.split_factor = options.split_factor
        self.em_type = options.em_type
        self.trans_counter = None
        self.lex_counter = None
        self.specify_splitting = options.specify_splitting
        self.function_mask = options.function_mask
        self.use_neural = options.use_neural
        self.unified_network = options.unified_network

        p_counter = 0
        for p in pos.keys():
            self.id_to_pos[self.pos[p]] = p
            if p == 'ROOT-POS':
                continue
            else:
                p_id = self.pos[p]
                self.decision_pos[p] = p_counter
                self.to_decision[p_id] = p_counter
                self.from_decision[p_counter] = p_id
                p_counter += 1
        self.dvalency = options.d_valency
        self.cvalency = options.c_valency
        # head_pos,child_pos,head_subtag,child_sub_tag,direction,child_valence
        self.trans_param = np.zeros((len(pos), len(pos), self.tag_num, self.tag_num, 2, self.cvalency))
        # head_pos,head_subtag,direction,valence,decision
        self.decision_param = np.zeros((len(self.decision_pos), self.tag_num, 2, self.dvalency, 2))
        # tag,subtag,word
        if self.use_lex:
            self.lex_param = np.zeros((len(pos), self.tag_num, len(self.vocab)))
        else:
            self.lex_param = None

        self.trans_alpha = None
        self.lex_alpha = None
        self.use_prior = options.use_prior

        if self.specify_splitting:
            self.specify_list = set()
            self.specify_list.add("V")
            # self.specify_list.add("NN")

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

    # KM initialization
    def init_param(self, data):
        root_idx = self.pos['ROOT-POS']
        count_smoothing = 0.1
        norm_counter = np.zeros((len(self.decision_pos), 2, self.dvalency, 2))  # pos_tag,direction,valence,decision
        for sentence in data:
            word_num = sentence.size - 1
            change = np.zeros((word_num, 2))
            for i, entry in enumerate(sentence.entries):
                if i == 0:
                    continue
                pos_id = self.pos[entry.pos]
                self.trans_param[root_idx, pos_id, 0, :, 1, :] += 1. / word_num
            for j, m_entry in enumerate(sentence.entries):
                if j == 0:
                    continue
                child_sum = 0
                for i in range(sentence.size):
                    if i == 0:
                        continue
                    if i == j:
                        continue
                    child_sum += 1. / abs(i - j)
                if child_sum > 0:
                    scale = float(word_num - 1) / word_num * (1. / child_sum)
                else:
                    scale = 0
                m_pos = m_entry.pos
                m_word = m_entry.norm
                m_pos_id = self.pos[m_pos]
                m_word_id = self.vocab[m_word]
                if self.use_lex:
                    self.lex_param[m_pos_id, :, m_word_id] += 1.
                for i, h_entry in enumerate(sentence.entries):
                    if i == j:
                        continue
                    if i == 0:
                        continue
                    if j < i:
                        dir = 0
                    else:
                        dir = 1
                    span = abs(i - j)
                    h_pos = h_entry.pos
                    m_pos = m_entry.pos
                    word = m_entry.norm
                    h_pos_id = self.pos.get(h_pos)
                    m_pos_id = self.pos.get(m_pos)
                    word_id = self.vocab.get(word)
                    # if self.use_lex:
                    #     self.lex_param[m_pos_id, :, word_id] += 1. / span * scale
                    self.trans_param[h_pos_id, m_pos_id, :, :, dir, :] += 1. / span * scale
                    change[i - 1, dir] += 1. / span * scale
            self.update_decision(change, norm_counter, sentence.entries)
        self.trans_param += count_smoothing
        self.decision_param += count_smoothing
        if self.use_lex:
            self.lex_param += count_smoothing
        es = self.first_child_update(norm_counter)
        pr_first_kid = 0.9 * es
        norm_counter = norm_counter * pr_first_kid
        norm_counter = norm_counter.reshape((len(self.decision_pos), 1, 2, self.dvalency, 2))
        self.decision_param = self.decision_param + norm_counter
        self.trans_param[:, root_idx, :, :, :, :] = 0
        # self.trans_param[root_idx, :, :, :, 0, :] = 0
        trans_sum = np.sum(self.trans_param, axis=(1, 3)).reshape((len(self.pos), 1, self.tag_num, 1, 2, self.cvalency))
        decision_sum = np.sum(self.decision_param, axis=4).reshape(
            (len(self.decision_pos), self.tag_num, 2, self.dvalency, 1))

        self.trans_param = self.trans_param / trans_sum
        self.decision_param = self.decision_param / decision_sum
        if self.use_lex:
            lex_sum = np.sum(self.lex_param, axis=2).reshape(len(self.pos), self.tag_num, 1)
            self.lex_param = self.lex_param / lex_sum

    def update_decision(self, change, norm_counter, entries):
        word_num, _ = change.shape
        for i in range(word_num):
            pos_id = self.decision_pos.get(entries[i + 1].pos)
            for dir in range(2):
                if change[i, dir] > 0:
                    norm_counter[pos_id, dir, 0, 1] += 1
                    norm_counter[pos_id, dir, 1, 1] += -1
                    self.decision_param[pos_id, :, dir, 1, 1] += change[i, dir]

                    norm_counter[pos_id, dir, 0, 0] += -1
                    norm_counter[pos_id, dir, 1, 0] += 1
                    self.decision_param[pos_id, :, dir, 0, 0] += 1
                else:
                    self.decision_param[pos_id, :, dir, 0, 0] += 1

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

    def em_e(self, batch_pos, batch_words, batch_sen, trans_counter, decision_counter, lex_counter, em_type):
        batch_pos = np.array(batch_pos)
        batch_words = np.array(batch_words)
        if em_type == 'viterbi':
            batch_likelihood = self.run_viterbi_estep(batch_pos, batch_words, batch_sen, trans_counter,
                                                      decision_counter, lex_counter)
        elif em_type == 'em':
            batch_likelihood = self.run_em_estep(batch_pos, batch_words, batch_sen, trans_counter, decision_counter,
                                                 lex_counter)

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
        self.lex_counter = lex_counter
        return batch_likelihood

    def run_em_estep(self, batch_pos, batch_words, batch_sen, trans_counter, decision_counter, lex_counter):
        batch_score, batch_decision_score = self.evaluate_batch_score(batch_words, batch_pos)
        batch_score = np.array(batch_score)
        batch_decision_score = np.array(batch_decision_score)
        # if self.specified_splitting:
        # batch_score, batch_decision_score = self.mask_scores(batch_score, batch_decision_score, batch_pos)
        batch_score[:, :, 0, :, :, :] = -np.inf
        if self.tag_num > 1:
            batch_score[:, 0, :, 1:, :, :] = -np.inf
        if self.function_mask:
            batch_score = self.function_to_mask(batch_score, batch_pos)
        inside_complete_table, inside_incomplete_table, sentence_prob = \
            eisner_for_dmv.batch_inside(batch_score, batch_decision_score, self.dvalency, self.cvalency)
        outside_complete_table, outside_incomplete_table = \
            eisner_for_dmv.batch_outside(inside_complete_table, inside_incomplete_table, batch_score,
                                         batch_decision_score, self.dvalency, self.cvalency)
        batch_likelihood = self.update_pseudo_count(inside_incomplete_table, inside_complete_table, sentence_prob,
                                                    outside_incomplete_table, outside_complete_table, trans_counter,
                                                    decision_counter, lex_counter, batch_pos, batch_words)
        self.trans_counter = trans_counter
        self.lex_counter = lex_counter
        return batch_likelihood

    def evaluate_batch_score(self, batch_words, batch_pos):
        batch_size, sentence_length = batch_words.shape
        # batch,head,child,head_tag,child_tag
        scores = np.zeros((batch_size, sentence_length, sentence_length, self.tag_num, self.tag_num, self.cvalency))
        # batch,position,tag,direction,valency,decision
        decision_scores = np.zeros((batch_size, sentence_length, self.tag_num, 2, self.dvalency, 2))
        # batch,position,tag,word
        lex_scores = np.zeros((batch_size, sentence_length, self.tag_num))
        scores.fill(-np.inf)
        decision_scores.fill(-np.inf)
        lex_scores.fill(-np.inf)
        for sentence_id in range(batch_size):
            for i in range(sentence_length):
                pos_id = batch_pos[sentence_id][i]
                if i > 0:
                    dec_pos_id = self.to_decision[pos_id]
                    decision_scores[sentence_id, i, :, :, :, :] = np.log(self.decision_param[dec_pos_id, :, :, :, :])
                else:
                    decision_scores[sentence_id, i, :, :, :, :] = 0
                for j in range(sentence_length):
                    h_pos_id = batch_pos[sentence_id][i]
                    m_pos_id = batch_pos[sentence_id][j]
                    m_word_id = batch_words[sentence_id][j]
                    if j == 0:
                        continue
                    if i == j:
                        continue
                    if i > j:
                        dir = 0
                    else:
                        dir = 1
                    scores[sentence_id, i, j, :, :, :] = np.log(self.trans_param[h_pos_id, m_pos_id, :, :, dir, :])
                    if self.use_lex:
                        lex_scores[sentence_id, j, :] = self.lex_param[m_pos_id, :, m_word_id]
                        scores[sentence_id, i, j, :, :, :] += np.log(
                            lex_scores[sentence_id, j, :].reshape(1, self.tag_num, 1))
        return scores, decision_scores

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

    def em_m(self, trans_counter, decision_counter, lex_counter, child_model, decision_model):
        root_idx = self.pos['ROOT-POS']
        if self.tag_num > 1:
            self.param_smoothing = 1e-8
        trans_counter = trans_counter + self.param_smoothing
        decision_counter = decision_counter + self.param_smoothing
        if self.use_lex:
            lex_counter = lex_counter + self.param_smoothing
        trans_counter[:, root_idx, :, :, :, :] = 0
        child_sum = np.sum(trans_counter, axis=(1, 3)).reshape(len(self.pos), 1, self.tag_num, 1, 2, self.cvalency)
        decision_sum = np.sum(decision_counter, axis=4).reshape(len(self.decision_pos), self.tag_num,
                                                                2, self.dvalency, 1)
        self.trans_param = trans_counter / child_sum
        self.decision_param = decision_counter / decision_sum
        if self.use_lex:
            lex_sum = np.sum(lex_counter, axis=2).reshape(len(self.pos), self.tag_num, 1)
            self.lex_param = lex_counter / lex_sum
        return

    def split_tags(self, trans_counter, prior_alpha, lex_counter, lex_prior_alpha):
        self.tag_num = self.tag_num * self.split_factor
        print "split maximum tag number from " + str(self.tag_num / self.split_factor) + " to " + str(self.tag_num)
        old_trans_param = np.copy(self.trans_param)
        old_decision_param = np.copy(self.decision_param)
        old_lex_param = np.copy(self.lex_param)

        self.trans_param = np.zeros((len(self.pos), len(self.pos), self.tag_num, self.tag_num, 2, self.cvalency))
        self.decision_param = np.zeros((len(self.decision_pos), self.tag_num, 2, self.dvalency, 2))
        self.lex_param = np.zeros((len(self.pos), self.tag_num, len(self.vocab)))
        self.trans_alpha = np.zeros((len(self.pos), 1, self.tag_num, 1, 2, self.cvalency))
        self.lex_alpha = np.zeros((len(self.pos), self.tag_num, len(self.vocab)))
        root_idx = self.pos["ROOT-POS"]

        for h_t in range(self.tag_num):
            decision_random = old_decision_param[:, h_t / self.split_factor, :, :, :] * (
                np.random.rand(len(self.decision_pos), 2, self.dvalency, 2) - 0.5) / 5000
            # self.decision_param[:, h_t, :, :, :] = old_decision_param[:, h_t / self.split_factor, :, :,
            #                                        :] + decision_random
            self.decision_param[:, h_t, :, :, :] = old_decision_param[:, h_t / self.split_factor, :, :, :]
            lex_random = old_lex_param[:, h_t / self.split_factor, :] * (
                np.random.rand(len(self.pos), len(self.vocab)) - 0.5) / 5000
            self.lex_param[:, h_t, :] = old_lex_param[:, h_t / self.split_factor, :] + lex_random
            # self.lex_param[:, h_t, :] = old_lex_param[:, h_t / 2, :]
            for m_t in range(self.tag_num):
                trans_random = old_trans_param[:, :, h_t / self.split_factor, m_t / self.split_factor,
                               :, :] / self.split_factor * (
                                   np.random.rand(len(self.pos), len(self.pos), 2, self.cvalency) - 0.5) / 5000
                # self.trans_param[:, :, h_t, m_t, :] = old_trans_param[:, :, h_t / self.split_factor,
                #                                       m_t / self.split_factor, :] / self.split_factor + trans_random
                self.trans_param[:, :, h_t, m_t, :, :] = old_trans_param[:, :, h_t / self.split_factor,
                                                         m_t / self.split_factor, :, :] / self.split_factor

        if self.use_prior:
            child_mean = np.average(trans_counter, axis=(1, 3)).reshape(len(self.pos), 1,
                                                                        self.tag_num / self.split_factor, 1, 2,
                                                                        self.cvalency)
            old_trans_alpha = -child_mean * prior_alpha
            for h_t in range(self.tag_num):
                self.trans_alpha[:, :, h_t, :, :, :] = old_trans_alpha[:, :, h_t / self.split_factor, :, :,
                                                       :] / self.split_factor

            lex_mean = np.average(lex_counter, axis=2).reshape(len(self.pos), self.tag_num / self.split_factor, 1)
            old_lex_alpha = -lex_mean * lex_prior_alpha
            for h_t in range(self.tag_num):
                self.lex_alpha[:, h_t, :] = old_lex_alpha[:, h_t / self.split_factor, :]

        lex_sum = np.sum(self.lex_param, axis=2).reshape(len(self.pos), self.tag_num, 1)
        self.lex_param = self.lex_param / lex_sum

    def specified_split_tags(self):
        self.tag_num = self.tag_num * self.split_factor
        print "split maximum tag number from " + str(self.tag_num / self.split_factor) + " to " + str(self.tag_num)
        old_trans_param = np.copy(self.trans_param)
        old_decision_param = np.copy(self.decision_param)
        old_lex_param = np.copy(self.lex_param)
        self.trans_param = np.zeros((len(self.pos), len(self.pos), self.tag_num, self.tag_num, 2, self.cvalency))
        self.decision_param = np.zeros((len(self.decision_pos), self.tag_num, 2, self.dvalency, 2))
        self.lex_param = np.zeros((len(self.pos), self.tag_num, len(self.vocab)))
        self.trans_alpha = np.zeros((len(self.pos), 1, self.tag_num, 1, 2, self.cvalency))
        self.lex_alpha = np.zeros((len(self.pos), self.tag_num, len(self.vocab)))

        for h_t in range(self.tag_num / self.split_factor):
            self.decision_param[:, h_t, :, :, :] = old_decision_param[:, h_t, :, :, :]
            self.lex_param[:, h_t, :] = old_lex_param[:, h_t, :]
            for m_t in range(self.tag_num / self.split_factor):
                self.trans_param[:, :, h_t, m_t, :, :] = old_trans_param[:, :, h_t, m_t, :, :]
        for s in self.specify_list:
            s_idx = self.pos[s]
            dec_sidx = self.to_decision[s_idx]
            for h_t in range(self.tag_num):
                self.decision_param[dec_sidx, h_t, :, :, :] = old_decision_param[dec_sidx, h_t / self.split_factor, :,
                                                              :, :]
                lex_random = old_lex_param[s_idx, h_t / self.split_factor, :] * (
                    np.random.rand(1, len(self.vocab)) - 0.5) / 5000
                self.lex_param[s_idx, h_t, :] = old_lex_param[s_idx, h_t / self.split_factor, :] + lex_random
                for m_t in range(self.tag_num / self.split_factor):
                    self.trans_param[s_idx, :, h_t, m_t, :, :] = old_trans_param[s_idx, :, h_t / self.split_factor, m_t,
                                                                 :, :]
                    self.trans_param[:, s_idx, m_t, h_t, :, :] = old_trans_param[:, s_idx, m_t, h_t / self.split_factor,
                                                                 :, :]
        for s in self.specify_list:
            s_idx = self.pos[s]
            for cs in self.specify_list:
                cs_idx = self.pos[cs]
                for h_t in range(h_t):
                    for m_t in range(m_t):
                        self.trans_param[s_idx, cs_idx, h_t, m_t, :, :] = old_trans_param[s_idx, cs_idx,
                                                                          h_t / self.split_factor,
                                                                          m_t / self.split_factor, :, :]
                        # lex_sum = np.sum(self.lex_param, axis=2).reshape(len(self.pos), self.tag_num, 1)
                        # self.lex_param = self.lex_param / lex_sum

    def update_pseudo_count(self, inside_incomplete_table, inside_complete_table, sentence_prob,
                            outside_incomplete_table, outside_complete_table, trans_counter,
                            decision_counter, lex_counter, batch_pos, batch_words):
        batch_likelihood = 0.0
        batch_size, sentence_length = batch_pos.shape
        span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(sentence_length, False)
        for sen_id in range(batch_size):
            pos_sentence = batch_pos[sen_id]
            word_sentence = batch_words[sen_id]
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
                    m_dec_pos = self.to_decision[m_pos]
                    m_word = word_sentence[m]
                    if dir == 0:
                        span_id = span_2_id[(m, h, dir)]
                    else:
                        span_id = span_2_id[(h, m, dir)]
                    dep_count = inside_incomplete_table[sen_id, span_id, :, :, :] + \
                                outside_incomplete_table[sen_id, span_id, :, :, :] - sentence_prob[sen_id]
                    if dir == 0:
                        dep_count = dep_count.swapaxes(1, 0)
                    if self.cvalency == 1:
                        trans_counter[h_pos, m_pos, :, :, dir, 0] += np.sum(np.exp(dep_count), axis=2)
                    else:
                        trans_counter[h_pos, m_pos, :, :, dir, :] += np.exp(dep_count)
                    if h > 0:
                        h_dec_pos = self.to_decision[h_pos]
                        decision_counter[h_dec_pos, :, dir, :, 1] += np.sum(np.exp(dep_count), axis=1)
                    if self.use_lex:
                        lex_counter[m_pos, :, m_word] += np.sum(np.exp(dep_count), axis=(0, 2))
            for m in range(1, sentence_length):
                m_pos = pos_sentence[m]
                m_dec_pos = self.to_decision[m_pos]
                m_word = word_sentence[m]
                for d in range(2):
                    m_span_id = span_2_id[(m, m, d)]
                    stop_count = inside_complete_table[sen_id, m_span_id, :, :] + \
                                 outside_complete_table[sen_id, m_span_id, :, :] - sentence_prob[sen_id]
                    decision_counter[m_dec_pos, :, d, :, 0] += np.exp(stop_count)
            batch_likelihood += sentence_prob[sen_id]
        return batch_likelihood

    def mask_scores(self, batch_scores, batch_decision_scores, batch_pos):
        batch_size, sentence_length, _, _, _, _ = batch_scores.shape
        score_mask = np.zeros((batch_size, sentence_length, sentence_length, self.tag_num, self.tag_num, self.cvalency))
        decision_mask = np.zeros((batch_size, sentence_length, self.tag_num, 2, self.dvalency, 2))
        for s in range(batch_size):
            for i in range(sentence_length):
                if self.id_to_pos[batch_pos[s, i]] not in self.specify_list:
                    decision_mask[s, i, 1:, :] = -np.inf
                for j in range(sentence_length):
                    if self.id_to_pos[batch_pos[s, i]] not in self.specify_list and self.id_to_pos[
                        batch_pos[s, j]] not in self.specify_list:
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
        batch_size, sentence_length, _, _, _, _ = batch_score.shape
        function_score_mask = np.zeros(
            (batch_size, sentence_length, sentence_length, self.tag_num, self.tag_num, self.cvalency))
        for s in range(batch_size):
            for i in range(sentence_length):
                pos_id = batch_pos[s, i]
                pos = self.id_to_pos[pos_id]
                if pos in self.function_set:
                    function_score_mask[s, i, :, :, :] = -np.inf
        batch_score = batch_score + function_score_mask
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
