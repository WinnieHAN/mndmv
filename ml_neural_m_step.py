import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import utils
import numpy as np
from numpy import linalg as LA
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class m_step_model(nn.Module):
    def __init__(self, tag_num, lan_num, options):
        super(m_step_model, self).__init__()
        self.tag_num = tag_num
        self.options = options
        self.cvalency = options.c_valency
        self.dvalency = options.d_valency
        self.drop_out = options.drop_out
        self.child_only = options.child_only
        self.gpu = options.gpu
        self.pembedding_dim = options.pembedding_dim
        self.valency_dim = options.valency_dim
        self.hid_dim = options.hid_dim
        self.pre_output_dim = options.pre_output_dim
        self.unified_network = options.unified_network
        self.decision_pre_output_dim = options.decision_pre_output_dim
        self.drop_out = options.drop_out
        self.lan_num = lan_num
        self.ml_comb_type = options.ml_comb_type  # options.ml_comb_type = 0(no_lang_id)/1(id embeddings)/2(classify-tags)
        self.stc_model_type = options.stc_model_type  # 1  lstm   2 lstm with atten   3 variational
        if self.ml_comb_type == 1:
            self.lang_dim = options.lang_dim  # options.lang_dim = 10(default)

        elif self.ml_comb_type == 2:
            self.lang_dim = options.lang_dim
            self.lstm_layer_num = options.lstm_layer_num  # 1
            self.lstm_hidden_dim = options.lstm_hidden_dim  # 10
            self.bidirectional = options.bidirectional  # True
            self.lstm_direct = 2 if self.bidirectional else 1
            self.hidden = self.init_hidden(1)  # 1 here is just for init, will be changed in forward process
            self.lstm = nn.LSTM(self.pembedding_dim, self.lstm_hidden_dim, num_layers=self.lstm_layer_num,
                                bidirectional=self.bidirectional,
                                batch_first=True)  # hidden_dim // 2, num_layers=1, bidirectional=True
            self.lang_classifier = nn.Linear(self.lstm_direct * self.lstm_hidden_dim, self.lan_num)
            if self.stc_model_type == 2:
                self.max_length = 40  # TODO, we train or test on len40
                self.nhid = self.pembedding_dim # for attention
                self.hvds_dim = self.nhid + self.lstm_hidden_dim * self.lstm_direct * self.lstm_layer_num  # for sts
                self.linear_hvds = nn.Linear(self.hvds_dim, self.max_length)

            if self.stc_model_type == 3:
                self.variational_mu = nn.Linear(self.lstm_hidden_dim * self.lstm_direct * self.lstm_layer_num, self.lstm_direct * self.lstm_hidden_dim)
                self.variational_logvar = nn.Linear(self.lstm_hidden_dim * self.lstm_direct * self.lstm_layer_num,
                                                    self.lstm_direct * self.lstm_hidden_dim)  # log var.pow(2)
        self.plookup = nn.Embedding(self.tag_num, self.pembedding_dim)
        self.dplookup = nn.Embedding(self.tag_num, self.pembedding_dim)
        self.vlookup = nn.Embedding(self.cvalency, self.valency_dim)
        self.dvlookup = nn.Embedding(self.dvalency, self.valency_dim)
        self.head_lstm_embeddings = self.plookup
        if self.ml_comb_type == 1:
            self.llookup = nn.Embedding(self.lan_num, self.lang_dim)

        self.dropout_layer = nn.Dropout(p=self.drop_out)

        self.dir_embed = options.dir_embed
        self.dir_dim = options.dir_dim
        if self.dir_embed:
            self.dlookup = nn.Embedding(2, self.dir_dim)
        if not self.dir_embed:
            if self.ml_comb_type == 0:
                self.left_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
                self.right_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
            elif self.ml_comb_type == 1:
                self.left_hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.lang_dim), self.hid_dim)
                self.right_hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.lang_dim), self.hid_dim)
            elif self.ml_comb_type == 2:
                self.left_hid = nn.Linear(
                    (self.pembedding_dim + self.valency_dim + self.lstm_direct * self.lstm_hidden_dim), self.hid_dim)
                self.right_hid = nn.Linear(
                    (self.pembedding_dim + self.valency_dim + self.lstm_direct * self.lstm_hidden_dim),
                    self.hid_dim)


        else:
            self.hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.dir_dim), self.hid_dim)
        self.linear_chd_hid = nn.Linear(self.hid_dim, self.pre_output_dim)
        self.pre_output = nn.Linear(self.pre_output_dim, self.tag_num)

        if not self.dir_embed:
            self.left_decision_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
            self.right_decision_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
        else:
            self.decision_hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.dir_dim), self.hid_dim)
        self.linear_decision_hid = nn.Linear(self.hid_dim, self.decision_pre_output_dim)
        self.decision_pre_output = nn.Linear(self.decision_pre_output_dim, 2)
        # self.decision_pre_output = nn.Linear(self.hid_dim, 2)
        self.em_type = options.em_type
        self.param_smoothing = options.param_smoothing

        self.optim_type = options.optim_type
        self.lr = options.learning_rate
        if self.optim_type == 'sgd':
            self.optim = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optim_type == 'adam':
            self.optim = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim_type == 'adagrad':
            self.optim = optim.Adagrad(self.parameters(), lr=self.lr)

    def init_hidden(self, batch_init):
        return (
            torch.zeros(self.lstm_layer_num * self.lstm_direct, batch_init, self.lstm_hidden_dim),
            # num_layers * bi-direction
            torch.zeros(2 * 1, batch_init, self.lstm_hidden_dim))

    def reparameterize(self, training, mu, logvar):
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.autograd.Variable(torch.randn(std.size()), requires_grad=False)  # torch.randn_like(std)
            temp = torch.mul(std, eps) + mu
            return temp  # eps.mul(std).add_(mu)
        else:
            return mu

    def lang_loss(self, stc_representation, batch_target):

        loss = torch.nn.CrossEntropyLoss()
        lang_output = self.lang_classifier(stc_representation)
        lang_loss = loss(lang_output, batch_target)
        return lang_loss


    def stc_representation(self, sentences, sentences_len, hid_tensor):
        batch_size = len(sentences)
        sentences_maxlen = len(sentences[0])
        if self.stc_model_type == 1:
            embeds = self.head_lstm_embeddings(sentences)
            lstm_out, self.hidden = self.lstm(embeds)
            sentences_all_lstm = torch.transpose(self.hidden[0], 0, 1)
            sentences_all_lstm = sentences_all_lstm.contiguous().view(sentences_all_lstm.size()[0], -1)
            return self.dropout_layer(sentences_all_lstm)
        elif self.stc_model_type == 2:
            # self.hidden = self.init_hidden(batch_size)  # sts batch
            embeds = self.head_lstm_embeddings(torch.autograd.Variable(torch.LongTensor(sentences)))
            # sts_packed = torch.nn.utils.rnn.PackedSequence(embeds, batch_sizes=sentences_len)
            # sentence_in = pad_packed_sequence(sts_packed, batch_first=True)
            # lstm_out, self.hidden = self.lstm(sentence_in[0], self.hidden)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
            lstm_out, self.hidden = self.lstm(embeds)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
            sentences_lstm = torch.transpose(self.hidden[0], 0, 1).contiguous().view(batch_size,
                                                                                     -1)  # batch_size* (num_layer*direct*hiddensize) #use h not c
            # sentences_all_lstm = torch.transpose(lstm_out, 0, 1)
            atten_weight = F.softmax(self.linear_hvds(torch.cat((hid_tensor, sentences_lstm), 1)))[:, 0:sentences_maxlen]
            attn_applied = torch.bmm(torch.transpose(atten_weight.unsqueeze(2), 1, 2), lstm_out)  # 1*1*6
            return attn_applied.squeeze(1)
        elif self.stc_model_type == 3:
            # self.hidden = self.init_hidden(batch_size)  # sts batch
            embeds = self.dropout_layer(self.head_lstm_embeddings(torch.autograd.Variable(torch.LongTensor(sentences))))
            # sts_packed = torch.nn.utils.rnn.PackedSequence(embeds, sentences_len)  # batch_sizes=
            # sentence_in = pad_packed_sequence(sts_packed, batch_first=True)
            # lstm_out, self.hidden = self.lstm(sentence_in[0], self.hidden)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
            lstm_out, self.hidden = self.lstm(embeds)  # [0]# sentence_in.view(BATCH_SIZE, BATCH_SIZE, -1)
            sentences_all_lstm = torch.transpose(self.hidden[0], 0, 1)
            sentences_all_lstm = sentences_all_lstm.contiguous().view(sentences_all_lstm.size()[0], -1)
            mu = self.variational_mu(sentences_all_lstm)
            logvar = self.variational_logvar(sentences_all_lstm)
            var_out = self.reparameterize(self.training, mu, logvar)
            return var_out, -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward_(self, batch_pos, batch_dir, batch_valence, batch_target, batch_target_count,
                 is_prediction, type, em_type, batch_lang_id, sentences, sentences_len):
        p_embeds = self.plookup(batch_pos)
        if type == 'child':
            v_embeds = self.vlookup(batch_valence)
        else:
            v_embeds = self.dvlookup(batch_valence)
        if self.dir_embed:
            d_embeds = self.dlookup(batch_dir)
        if not self.dir_embed:
            left_mask, right_mask = self.construct_mask(batch_dir)
            if self.ml_comb_type == 0:
                input_embeds = torch.cat((p_embeds, v_embeds), 1)
            elif self.ml_comb_type == 1:
                lang_embeds = self.llookup(batch_lang_id)
                input_embeds = torch.cat((p_embeds, v_embeds, lang_embeds), 1)
            elif self.ml_comb_type == 2:
                stc_representation_and_vae_loss = self.stc_representation(sentences, sentences_len, p_embeds)
                stc_representation = stc_representation_and_vae_loss[0] if isinstance(stc_representation_and_vae_loss,
                                                                                      tuple) else stc_representation_and_vae_loss
                vae_loss = stc_representation_and_vae_loss[1] if isinstance(stc_representation_and_vae_loss,
                                                                            tuple) else 0
                input_embeds = torch.cat((p_embeds, v_embeds, stc_representation), 1)
                if not is_prediction:
                    lang_cls_loss = self.lang_loss(stc_representation, batch_lang_id)
            input_embeds = self.dropout_layer(input_embeds)
            left_v = self.left_hid(input_embeds)
            left_v = F.relu(left_v)
            right_v = self.right_hid(input_embeds)
            right_v = F.relu(right_v)
            left_v = left_v.masked_fill(left_mask, 0.0)
            right_v = right_v.masked_fill(right_mask, 0.0)
            hidden_v = left_v + right_v
        else:
            input_embeds = torch.cat((p_embeds, v_embeds, d_embeds), 1)
            hidden_v = self.hid(input_embeds)
        if type == 'child':
            pre_output_v = self.pre_output(F.relu(self.linear_chd_hid(hidden_v)))
        else:
            pre_output_v = self.decision_pre_output(F.relu(self.linear_decision_hid(hidden_v)))
        if not is_prediction:
            if em_type == 'viterbi':
                loss = torch.nn.CrossEntropyLoss()
                batch_loss = loss(pre_output_v, batch_target)
                return batch_loss
            else:
                predicted_prob = F.log_softmax(pre_output_v, dim=1)
                batch_target = batch_target.view(len(batch_target), 1)
                target_prob = torch.gather(predicted_prob, 1, batch_target)
                batch_target_count = batch_target_count.view(len(batch_target_count), 1)
                batch_loss = -torch.sum(batch_target_count * target_prob)
                if self.ml_comb_type == 2:
                    batch_loss += lang_cls_loss
                if self.stc_model_type == 3:
                    batch_loss += vae_loss
                return batch_loss
        else:
            predicted_param = F.softmax(pre_output_v, dim=1)
            return predicted_param

    def forward_decision(self, batch_decision_pos, batch_decision_dir, batch_dvalence, batch_target_decision,
                         batch_target_decision_count, is_prediction, em_type):
        p_embeds = self.dplookup(batch_decision_pos)
        v_embeds = self.dvlookup(batch_dvalence)

        if self.dir_embed:
            d_embeds = self.dlookup(batch_decision_dir)
        if not self.dir_embed:

            left_mask, right_mask = self.construct_mask(batch_decision_dir)
            input_embeds = torch.cat((p_embeds, v_embeds), 1)

            left_v = self.left_decision_hid(input_embeds)
            left_v = F.relu(left_v)
            right_v = self.right_decision_hid(input_embeds)
            right_v = F.relu(right_v)
            left_v = left_v.masked_fill(left_mask, 0.0)
            right_v = right_v.masked_fill(right_mask, 0.0)
            hidden_v = left_v + right_v
        else:
            input_embeds = torch.cat((p_embeds, v_embeds, d_embeds), 1)
            hidden_v = self.decision_hid(input_embeds)
        pre_output_v = self.decision_pre_output(F.relu(self.linear_decision_hid(hidden_v)))
        if not is_prediction:
            if em_type == 'viterbi':
                loss = torch.nn.CrossEntropyLoss()
                batch_loss = loss(pre_output_v, batch_target_decision)
                return batch_loss
            else:
                predicted_prob = F.log_softmax(pre_output_v, dim=1)
                batch_target = batch_target_decision.view(len(batch_target_decision), 1)
                target_prob = torch.gather(predicted_prob, 1, batch_target)
                batch_target_count = batch_target_decision_count.view(len(batch_target_decision_count), 1)
                batch_loss = -torch.sum(batch_target_count * target_prob)
                return batch_loss
        else:
            predicted_param = F.softmax(pre_output_v, dim=1)
            return predicted_param

    def construct_mask(self, batch_zero_one):
        batch_size = len(batch_zero_one)
        left_compare = torch.ones(batch_size, dtype=torch.long)
        right_compare = torch.zeros(batch_size, dtype=torch.long)
        left_mask = torch.eq(batch_zero_one, left_compare)
        right_mask = torch.eq(batch_zero_one, right_compare)
        left_mask = left_mask.view(batch_size, 1)
        right_mask = right_mask.view(batch_size, 1)
        left_mask = left_mask.expand(-1, self.hid_dim)
        right_mask = right_mask.expand(-1, self.hid_dim)
        return left_mask, right_mask

    def predict(self, sentence_trans_param, root_param, decision_param, batch_size, root_counnter, decision_counter,
                child_only, sentence_map, language_map, languages):
        _, input_pos_num, target_pos_num, dir_num, cvalency = sentence_trans_param.shape
        input_decision_pos_num, decision_dir_num, dvalency, target_decision_num = decision_param.shape
        input_trans_list = [[p, d, cv] for p in range(input_pos_num) for d in range(dir_num) for cv in range(cvalency)]
        input_decision_list = [[p, d, dv] for p in range(input_decision_pos_num) for d in range(dir_num) for dv in
                               range(dvalency)]

        batched_input_trans = utils.construct_update_batch_data(input_trans_list, batch_size)
        batched_input_decision = utils.construct_update_batch_data(input_decision_list, batch_size)
        trans_batch_num = len(batched_input_trans)
        decision_batch_num = len(batched_input_decision)
        for s in range(len(sentence_map)):
            for i in range(trans_batch_num):
                # Update transition parameters
                one_batch_size = len(batched_input_trans[i])
                batch_target_lan_v = torch.LongTensor([languages[language_map[s]]]).expand(one_batch_size)  # TODO hanwj
                batch_input_len = torch.LongTensor([len(sentence_map[s])]).expand(one_batch_size)
                batch_input_sen_v = torch.LongTensor([sentence_map[s]]).expand(one_batch_size, len(sentence_map[s]))
                one_batch_input_pos = torch.LongTensor(batched_input_trans[i])[:, 0]
                one_batch_dir = torch.LongTensor(batched_input_trans[i])[:, 1]
                one_batch_cvalency = torch.LongTensor(batched_input_trans[i])[:, 2]
                one_batch_input_pos_index = np.array(batched_input_trans[i])[:, 0]
                one_batch_dir_index = np.array(batched_input_trans[i])[:, 1]
                one_batch_cvalency_index = np.array(batched_input_trans[i])[:, 2]
                predicted_trans_param = self.forward_(one_batch_input_pos, one_batch_dir, one_batch_cvalency,
                                                      None, None, True, 'child',
                                                      self.em_type, batch_target_lan_v, batch_input_sen_v,
                                                      batch_input_len)
                sentence_trans_param[s][one_batch_input_pos_index, :, one_batch_dir_index, \
                one_batch_cvalency_index] = predicted_trans_param.detach().numpy()#.reshape(one_batch_size, target_pos_num, 1, 1)
        # TODO:
        # if not child_only:
        #     for i in range(decision_batch_num):
        #         # Update decision parameters
        #         one_batch_size = len(batched_input_decision[i])
        #         if self.unified_network:
        #             one_batch_input_decision_pos = torch.LongTensor(
        #                 map(lambda p: from_decision[p], np.array(batched_input_decision[i])[:, 0]))
        #         else:
        #             one_batch_input_decision_pos = torch.LongTensor(batched_input_decision[i])[:, 0]
        #         one_batch_decision_dir = torch.LongTensor(batched_input_decision[i])[:, 1]
        #         one_batch_dvalency = torch.LongTensor(batched_input_decision[i])[:, 2]
        #         if self.unified_network:
        #             one_batch_input_decision_pos_index = np.array(one_batch_input_decision_pos).tolist()
        #             one_batch_input_decision_pos_index = np.array(
        #                 map(lambda p: to_decision[p], one_batch_input_decision_pos_index))
        #         else:
        #             one_batch_input_decision_pos_index = np.array(batched_input_decision[i])[:, 0]
        #         one_batch_decision_dir_index = np.array(batched_input_decision[i])[:, 1]
        #         one_batch_dvalency_index = np.array(batched_input_decision[i])[:, 2]
        #         if self.unified_network:
        #             predicted_decision_param = self.forward_(one_batch_input_decision_pos, one_batch_decision_dir,
        #                                                      one_batch_dvalency, None, None, True, 'decision',
        #                                                      self.em_type)
        #         else:
        #             predicted_decision_param = self.forward_decision(one_batch_input_decision_pos,
        #                                                              one_batch_decision_dir, one_batch_dvalency,
        #                                                              None, None, True, self.em_type)
        #         decision_param[one_batch_input_decision_pos_index, :, one_batch_decision_dir_index,
        #         one_batch_dvalency_index, :] = predicted_decision_param.detach().numpy().reshape(one_batch_size, 1,
        #                                                                                          target_decision_num)
        if child_only:
            decision_counter = decision_counter + self.param_smoothing
            decision_sum = np.sum(decision_counter, axis=3, keepdims=True)
            decision_param = decision_counter / decision_sum

            root_counnter = root_counnter + self.param_smoothing
            root_sum = np.sum(root_counnter)
            root_param = root_counnter / root_sum


        # decision_counter = decision_counter + self.param_smoothing
        # decision_sum = np.sum(decision_counter, axis=3, keepdims=True)
        # decision_param_compare = decision_counter / decision_sum
        # decision_difference = decision_param_compare - decision_param
        # if not self.child_only:
        #     print 'distance for decision in this iteration ' + str(LA.norm(decision_difference))
        # trans_counter = trans_counter + self.param_smoothing
        # child_sum = np.sum(trans_counter, axis=(1, 3), keepdims=True)
        # trans_param_compare = trans_counter / child_sum
        # trans_difference = trans_param_compare - trans_param
        # print 'distance for trans in this iteration ' + str(LA.norm(trans_difference))
        return sentence_trans_param, root_param, decision_param
