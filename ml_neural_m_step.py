import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import utils
import numpy as np
import sys
from tqdm import tqdm
from numpy import linalg as LA
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class m_step_model(nn.Module):
    def __init__(self, tag_num, lan_num, sentence_map, language_map, languages, options):
        super(m_step_model, self).__init__()
        self.tag_num = tag_num
        self.options = options
        self.cvalency = options.c_valency
        self.dvalency = options.d_valency
        self.drop_out = options.drop_out
        self.child_only = options.child_only
        self.neural_epoch = options.neural_epoch
        self.gpu = options.gpu
        self.pembedding_dim = options.pembedding_dim
        self.valency_dim = options.valency_dim
        self.hid_dim = options.hid_dim
        self.pre_output_dim = options.pre_output_dim
        self.unified_network = options.unified_network
        self.decision_pre_output_dim = options.decision_pre_output_dim
        self.drop_out = options.drop_out
        self.lan_num = lan_num
        self.sample_batch_size = options.sample_batch_size
        self.sentence_map = sentence_map
        self.language_map = language_map
        self.languages = languages
        self.embed_languages = options.embed_languages
        self.sentence_predict = options.sentence_predict
        self.language_predict = options.language_predict
        self.stc_model_type = 1  # 1  lstm   2 lstm with atten   3 variational
        self.lan_dim = options.lan_dim  # options.lang_dim = 10(default)

        self.param_predict = False
        if self.sentence_predict or self.language_predict:
            self.lstm_layer_num = options.lstm_layer_num  # 1
            self.lstm_hidden_dim = options.lstm_hidden_dim  # 10
            self.bidirectional = options.bidirectional  # True
            self.lstm_direct = 2 if self.bidirectional else 1
            self.lstm = nn.LSTM(self.pembedding_dim, self.lstm_hidden_dim, num_layers=self.lstm_layer_num,
                                bidirectional=self.bidirectional, batch_first=True)
            self.sentence_mlp = nn.Linear(self.lstm_direct * self.lstm_hidden_dim, self.hid_dim)
            self.lan_classifier_mlp = nn.Linear(self.hid_dim, self.lan_dim)
        self.plookup = nn.Embedding(self.tag_num, self.pembedding_dim)
        self.dplookup = nn.Embedding(self.tag_num, self.pembedding_dim)
        self.vlookup = nn.Embedding(self.cvalency, self.valency_dim)
        self.dvlookup = nn.Embedding(self.dvalency, self.valency_dim)
        self.head_lstm_embeddings = self.plookup
        if self.embed_languages:
            self.llookup = nn.Embedding(self.lan_num, self.lan_dim)
        self.dropout_layer = nn.Dropout(p=self.drop_out)

        if self.embed_languages:
            self.left_hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.lan_dim), self.hid_dim)
            self.right_hid = nn.Linear((self.pembedding_dim + self.valency_dim + self.lan_dim), self.hid_dim)
        elif self.sentence_predict:
            self.left_hid = nn.Linear(
                (self.pembedding_dim + self.valency_dim + self.lstm_direct * self.lstm_hidden_dim), self.hid_dim)
            self.right_hid = nn.Linear(
                (self.pembedding_dim + self.valency_dim + self.lstm_direct * self.lstm_hidden_dim), self.hid_dim)
        else:
            self.left_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
            self.right_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
        self.linear_chd_hid = nn.Linear(self.hid_dim, self.pre_output_dim)
        self.pre_output = nn.Linear(self.pre_output_dim, self.tag_num)

        self.left_decision_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
        self.right_decision_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)

        self.linear_decision_hid = nn.Linear(self.hid_dim, self.decision_pre_output_dim)
        self.decision_pre_output = nn.Linear(self.decision_pre_output_dim, 2)

        self.lan_pre_output = nn.Linear(self.lan_dim, self.lan_num)
        self.lan_pre_output.weight = self.llookup.weight
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

    def lan_loss(self, sentence_v, batch_target, is_prediction):
        lan_output = self.lan_classifier_mlp(sentence_v)
        lan_output = self.lan_pre_output(lan_output)
        lan_output = F.log_softmax(lan_output, dim=1)
        batch_target = batch_target.view(len(batch_target), 1)
        target_prob = torch.gather(lan_output, 1, batch_target)
        _, lan_prediction = torch.max(lan_output, 1)
        if not is_prediction:
            #loss = torch.nn.NLLLoss()
            #lang_loss = loss(lan_output, batch_target.view(-1))
            lan_loss = -torch.sum(target_prob)
        else:
            lan_loss = None
        return lan_loss, lan_prediction

    def get_sentence_v(self, sentences):
        if sentences is not None:
            embeds = self.head_lstm_embeddings(sentences)
            lstm_out, self.hidden = self.lstm(embeds)
            # sentences_lstm = torch.transpose(self.hidden[0], 0, 1)
            # sentences_lstm = sentences_lstm.contiguous().view(sentences_lstm.size()[0], -1)
            sentences_lstm = torch.mean(lstm_out, dim=1)
            sentence_v = self.dropout_layer(sentences_lstm)
            sentence_v = self.sentence_mlp(sentence_v)
            sentence_v = F.relu(sentence_v)
        else:
            sentence_v = None
        return sentence_v

    def forward_(self, batch_pos, batch_dir, batch_valence, batch_target, batch_target_count,
                 is_prediction, type, batch_lan_id, sentences, sentences_len):
        p_embeds = self.plookup(batch_pos)
        if type == 'child':
            v_embeds = self.vlookup(batch_valence)
        else:
            v_embeds = self.dvlookup(batch_valence)
        left_mask, right_mask = self.construct_mask(batch_dir)
        lan_loss = torch.tensor([0.0])
        lan_prediction = torch.zeros(len(batch_pos))
        input_embeds = torch.cat((p_embeds, v_embeds), 1)
        if self.embed_languages:
            lan_embeds = self.llookup(batch_lan_id)
            input_embeds = torch.cat((input_embeds, lan_embeds), 1)
        if self.sentence_predict or self.language_predict:
            sentence_v = self.get_sentence_v(sentences)
            if self.sentence_predict:
                input_embeds = torch.cat((p_embeds, v_embeds, sentence_v), 1)
            if sentence_v is not None:
                lan_loss, lan_prediction = self.lan_loss(sentence_v, batch_lan_id, is_prediction)
        input_embeds = self.dropout_layer(input_embeds)
        left_v = self.left_hid(input_embeds)
        left_v = F.relu(left_v)
        right_v = self.right_hid(input_embeds)
        right_v = F.relu(right_v)
        left_v = left_v.masked_fill(left_mask, 0.0)
        right_v = right_v.masked_fill(right_mask, 0.0)
        hidden_v = left_v + right_v
        if type == 'child':
            pre_output_v = self.pre_output(F.relu(self.linear_chd_hid(hidden_v)))
        else:
            pre_output_v = self.decision_pre_output(F.relu(self.linear_decision_hid(hidden_v)))
        if not is_prediction:
            if self.em_type == 'viterbi':
                loss = torch.nn.CrossEntropyLoss()
                batch_loss = loss(pre_output_v, batch_target)
                return batch_loss
            else:
                predicted_prob = F.log_softmax(pre_output_v, dim=1)
                batch_target = batch_target.view(len(batch_target), 1)
                target_prob = torch.gather(predicted_prob, 1, batch_target)
                batch_target_count = batch_target_count.view(len(batch_target_count), 1)
                batch_loss = -torch.sum(batch_target_count * target_prob)
                if self.sentence_predict or self.language_predict:
                    batch_loss += lan_loss
                return batch_loss, lan_loss
        else:
            predicted_param = F.softmax(pre_output_v, dim=1)
            return predicted_param, lan_prediction

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

    def predict(self, trans_param, decision_param, decision_counter, child_only):
        self.eval()
        if self.sentence_predict:
            _, input_pos_num, target_pos_num, dir_num, cvalency = trans_param.shape
        else:
            input_pos_num, target_pos_num, dir_num, cvalency, lan_num = trans_param.shape
        input_decision_pos_num, decision_dir_num, dvalency, target_decision_num, lan_num = decision_param.shape
        input_trans_list = [[p, d, cv, l] for p in range(input_pos_num) for d in range(dir_num) for cv in
                            range(cvalency) for l in range(lan_num)]
        input_decision_list = [[p, d, dv, l] for p in range(input_decision_pos_num) for d in range(dir_num) for dv in
                               range(dvalency) for l in range(lan_num)]

        batched_input_trans = utils.construct_update_batch_data(input_trans_list, self.sample_batch_size)
        batched_input_decision = utils.construct_update_batch_data(input_decision_list, self.sample_batch_size)
        trans_batch_num = len(batched_input_trans)
        decision_batch_num = len(batched_input_decision)
        if self.sentence_predict:
            for s in range(len(self.sentence_map)):
                for i in range(trans_batch_num):
                    # Update transition parameters
                    one_batch_size = len(batched_input_trans[i])
                    batch_target_lan_v = torch.LongTensor([self.languages[self.language_map[s]]]).expand(
                        one_batch_size)
                    batch_input_len = torch.LongTensor([len(self.sentence_map[s])]).expand(one_batch_size)
                    batch_input_sen_v = torch.LongTensor([self.sentence_map[s]]).expand(one_batch_size,
                                                                                        len(self.sentence_map[s]))
                    one_batch_input_pos = torch.LongTensor(batched_input_trans[i])[:, 0]
                    one_batch_dir = torch.LongTensor(batched_input_trans[i])[:, 1]
                    one_batch_cvalency = torch.LongTensor(batched_input_trans[i])[:, 2]
                    # Parameter index for update
                    one_batch_input_pos_index = np.array(batched_input_trans[i])[:, 0]
                    one_batch_dir_index = np.array(batched_input_trans[i])[:, 1]
                    one_batch_cvalency_index = np.array(batched_input_trans[i])[:, 2]
                    predicted_trans_param, _ = self.forward_(one_batch_input_pos, one_batch_dir, one_batch_cvalency,
                                                             None, None, True, 'child', batch_target_lan_v,
                                                             batch_input_sen_v, batch_input_len)
                    trans_param[s][one_batch_input_pos_index, :, one_batch_dir_index,
                    one_batch_cvalency_index] = predicted_trans_param.detach().numpy()
        else:
            for i in range(trans_batch_num):
                one_batch_size = len(batched_input_trans[i])
                one_batch_input_pos = torch.LongTensor(batched_input_trans[i])[:, 0]
                one_batch_dir = torch.LongTensor(batched_input_trans[i])[:, 1]
                one_batch_cvalency = torch.LongTensor(batched_input_trans[i])[:, 2]
                one_batch_lan = torch.LongTensor(batched_input_trans[i])[:, 3]
                # Parameter index for update
                one_batch_input_pos_index = np.array(batched_input_trans[i])[:, 0]
                one_batch_dir_index = np.array(batched_input_trans[i])[:, 1]
                one_batch_cvalency_index = np.array(batched_input_trans[i])[:, 2]
                one_batch_lan_index = np.array(batched_input_trans[i])[:, 3]
                predicted_trans_param, _ = self.forward_(one_batch_input_pos, one_batch_dir, one_batch_cvalency,
                                                         None, None, True, 'child', one_batch_lan, None, None)
                trans_param[one_batch_input_pos_index, :, one_batch_dir_index, one_batch_cvalency_index,
                one_batch_lan_index] = predicted_trans_param.detach().numpy()

        if not child_only:
            for i in range(decision_batch_num):
                # Update decision parameters
                one_batch_input_decision_pos = torch.LongTensor(np.array(batched_input_decision[i])[:, 0])
                one_batch_decision_dir = torch.LongTensor(batched_input_decision[i])[:, 1]
                one_batch_dvalency = torch.LongTensor(batched_input_decision[i])[:, 2]
                one_batch_decision_lan = torch.LongTensor(batched_input_decision[i])[:, 3]
                # Decision parameter index for update
                one_batch_input_decision_pos_index = np.array(batched_input_decision[i])[:, 0]
                one_batch_decision_dir_index = np.array(batched_input_decision[i])[:, 1]
                one_batch_dvalency_index = np.array(batched_input_decision[i])[:, 2]
                one_batch_decision_lan_index = np.array(batched_input_decision[i])[:, 3]
                predicted_decision_param, _ = self.forward_(one_batch_input_decision_pos, one_batch_decision_dir,
                                                            one_batch_dvalency, None, None, True, 'decision',
                                                            one_batch_decision_lan, None, None)
                decision_param[one_batch_input_decision_pos_index, one_batch_decision_dir_index,
                one_batch_dvalency_index, :, one_batch_decision_lan_index] = predicted_decision_param.detach().numpy()
        else:
            decision_counter = decision_counter + self.param_smoothing
            decision_sum = np.sum(decision_counter, axis=3, keepdims=True)
            decision_param = decision_counter / decision_sum
        return trans_param, decision_param

    def batch_training(self, rule_samples, decision_samples, data_pos):
        self.train()
        self.param_predict = False
        for e in range(self.neural_epoch):
            iter_loss = 0.0
            iter_lang_loss = 0.0
            # Put training samples in batches
            batch_input_data, batch_target_data, batch_decision_data, batch_decision_target_data = \
                utils.construct_ml_input_data(rule_samples, decision_samples, self.sentence_map, self.sample_batch_size,
                                              self.em_type)
            # print 'batch_data for training constructed'
            batch_num = len(batch_input_data['input_pos'])
            tot_batch = batch_num

            # for batch_id in range(batch_num):
            for batch_id in tqdm(range(batch_num), mininterval=2,
                                 desc=' -Tot it %d (epoch %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
                # Input for the network: head_pos,direction, valency
                batch_input_pos_v = torch.LongTensor(batch_input_data['input_pos'][batch_id])
                batch_input_dir_v = torch.LongTensor(batch_input_data['input_dir'][batch_id])
                batch_cvalency_v = torch.LongTensor(batch_input_data['cvalency'][batch_id])
                # Input tensor for sentences
                if self.sentence_predict or self.language_predict:
                    batch_input_sen_v = []
                    for sentence_id in batch_input_data['sentence'][batch_id]:
                        batch_input_sen_v.append(data_pos[int(sentence_id)])
                    batch_input_sen_v = torch.LongTensor(batch_input_sen_v)
                    batch_input_len = len(batch_input_sen_v[0])
                else:
                    batch_input_sen_v = None
                    batch_input_len = None
                # Target for training: language and child POS
                batch_target_lan_v = torch.LongTensor(batch_target_data['target_lan'][batch_id])
                batch_target_pos_v = torch.LongTensor(batch_target_data['target_pos'][batch_id])
                if self.em_type == 'em':
                    batch_target_pos_count_v = torch.FloatTensor(batch_target_data['target_count'][batch_id])
                else:
                    batch_target_pos_count_v = None

                batch_loss, lang_loss = self.forward_(batch_input_pos_v, batch_input_dir_v, batch_cvalency_v,
                                                      batch_target_pos_v, batch_target_pos_count_v, False,
                                                      'child', batch_target_lan_v, batch_input_sen_v,
                                                      batch_input_len)
                iter_loss += batch_loss
                iter_lang_loss += lang_loss
                batch_loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            print "child loss for this iteration is " + str(iter_loss.detach().data.numpy() / batch_num)
            if self.sentence_predict or self.language_predict:
                print "language loss for this iteration is " + str(iter_lang_loss.detach().data.numpy() / batch_num)
            if not self.child_only:
                decision_batch_num = len(batch_decision_data['decision_pos'])
                tot_batch = decision_batch_num
                iter_decision_loss = 0.0
                for decision_batch_id in tqdm(
                        range(decision_batch_num), mininterval=2,
                        desc=' -Tot it %d (iter %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
                    # Input for decision network: pos,direction,decision_valency
                    batch_decision_pos_v = torch.LongTensor(batch_decision_data['decision_pos'][decision_batch_id])
                    batch_dvalency_v = torch.LongTensor(batch_decision_data['dvalency'][decision_batch_id])
                    batch_decision_dir_v = torch.LongTensor(batch_decision_data['decision_dir'][decision_batch_id])
                    batch_decision_lan_v = torch.LongTensor(batch_decision_data['decision_language'][decision_batch_id])
                    batch_target_decision_v = torch.LongTensor(
                        batch_decision_target_data['decision_target'][decision_batch_id])
                    if self.em_type == 'em':
                        batch_target_decision_count_v = torch.FloatTensor(
                            batch_decision_target_data['decision_target_count'][decision_batch_id])
                    else:
                        batch_target_decision_count_v = None
                    if self.sentence_predict:
                        batch_decision_sen_v = []
                        for sentence_id in batch_decision_data['sentence'][decision_batch_id]:
                            batch_decision_sen_v.append(data_pos[int(sentence_id)])
                        batch_decision_sen_v = torch.LongTensor(batch_decision_sen_v)
                        batch_decision_len = len(batch_decision_sen_v[0])
                    else:
                        batch_decision_sen_v = None
                        batch_decision_len = None
                    batch_decision_loss, _ = self.forward_(batch_decision_pos_v, batch_decision_dir_v,
                                                           batch_dvalency_v, batch_target_decision_v,
                                                           batch_target_decision_count_v, False, 'decision',
                                                           batch_decision_lan_v, batch_decision_sen_v,
                                                           batch_decision_len)
                    iter_decision_loss += batch_decision_loss
                    batch_decision_loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                print "decision loss for this iteration is " + str(
                    iter_decision_loss.detach().data.numpy() / batch_num)
