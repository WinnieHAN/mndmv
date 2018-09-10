import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import utils
import numpy as np
from numpy import linalg as LA


class m_step_model(nn.Module):
    def __init__(self, tag_num, options):
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

        self.plookup = nn.Embedding(self.tag_num, self.pembedding_dim)
        self.dplookup = nn.Embedding(self.tag_num, self.pembedding_dim)
        self.vlookup = nn.Embedding(self.cvalency, self.valency_dim)
        self.dvlookup = nn.Embedding(self.dvalency, self.valency_dim)

        self.dropout_layer = nn.Dropout(p=self.drop_out)

        self.dir_embed = options.dir_embed
        self.dir_dim = options.dir_dim
        if self.dir_embed:
            self.dlookup = nn.Embedding(2, self.dir_dim)
        if not self.dir_embed:
            self.left_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
            self.right_hid = nn.Linear((self.pembedding_dim + self.valency_dim), self.hid_dim)
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

    def forward_(self, batch_pos, batch_dir, batch_valence, batch_target, batch_target_count,
                 is_prediction, type, em_type):
        p_embeds = self.plookup(batch_pos)
        if type == 'child':
            v_embeds = self.vlookup(batch_valence)
        else:
            v_embeds = self.dvlookup(batch_valence)
        if self.dir_embed:
            d_embeds = self.dlookup(batch_dir)
        if not self.dir_embed:
            left_mask, right_mask = self.construct_mask(batch_dir)
            input_embeds = torch.cat((p_embeds, v_embeds), 1)
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

    def predict(self, trans_param, decision_param, batch_size, decision_counter, from_decision, to_decision,
                child_only,trans_counter):
        input_pos_num, target_pos_num, _, _, dir_num, cvalency = trans_param.shape
        input_decision_pos_num, _, decision_dir_num, dvalency, target_decision_num = decision_param.shape
        input_trans_list = [[p, d, cv] for p in range(input_pos_num) for d in range(dir_num) for cv in range(cvalency)]
        input_decision_list = [[p, d, dv] for p in range(input_decision_pos_num) for d in range(dir_num) for dv in
                               range(dvalency)]

        batched_input_trans = utils.construct_update_batch_data(input_trans_list, batch_size)
        batched_input_decision = utils.construct_update_batch_data(input_decision_list, batch_size)

        trans_batch_num = len(batched_input_trans)
        decision_batch_num = len(batched_input_decision)
        for i in range(trans_batch_num):
            # Update transition parameters
            one_batch_size = len(batched_input_trans[i])
            one_batch_input_pos = torch.LongTensor(batched_input_trans[i])[:, 0]
            one_batch_dir = torch.LongTensor(batched_input_trans[i])[:, 1]
            one_batch_cvalency = torch.LongTensor(batched_input_trans[i])[:, 2]
            one_batch_input_pos_index = np.array(batched_input_trans[i])[:, 0]
            one_batch_dir_index = np.array(batched_input_trans[i])[:, 1]
            one_batch_cvalency_index = np.array(batched_input_trans[i])[:, 2]
            predicted_trans_param = self.forward_(one_batch_input_pos, one_batch_dir, one_batch_cvalency, None, None,
                                                  True, 'child', self.em_type)
            trans_param[one_batch_input_pos_index, :, :, :, one_batch_dir_index,
            one_batch_cvalency_index] = predicted_trans_param.detach().numpy().reshape(one_batch_size, target_pos_num,
                                                                                       1, 1)
        if not child_only:
            for i in range(decision_batch_num):
                # Update decision parameters
                one_batch_size = len(batched_input_decision[i])
                if self.unified_network:
                    one_batch_input_decision_pos = torch.LongTensor(
                        map(lambda p: from_decision[p], np.array(batched_input_decision[i])[:, 0]))
                else:
                    one_batch_input_decision_pos = torch.LongTensor(batched_input_decision[i])[:, 0]
                one_batch_decision_dir = torch.LongTensor(batched_input_decision[i])[:, 1]
                one_batch_dvalency = torch.LongTensor(batched_input_decision[i])[:, 2]
                if self.unified_network:
                    one_batch_input_decision_pos_index = np.array(one_batch_input_decision_pos).tolist()
                    one_batch_input_decision_pos_index = np.array(
                        map(lambda p: to_decision[p], one_batch_input_decision_pos_index))
                else:
                    one_batch_input_decision_pos_index = np.array(batched_input_decision[i])[:, 0]
                one_batch_decision_dir_index = np.array(batched_input_decision[i])[:, 1]
                one_batch_dvalency_index = np.array(batched_input_decision[i])[:, 2]
                if self.unified_network:
                    predicted_decision_param = self.forward_(one_batch_input_decision_pos, one_batch_decision_dir,
                                                             one_batch_dvalency, None, None, True, 'decision',
                                                             self.em_type)
                else:
                    predicted_decision_param = self.forward_decision(one_batch_input_decision_pos,
                                                                     one_batch_decision_dir, one_batch_dvalency,
                                                                     None, None, True, self.em_type)
                decision_param[one_batch_input_decision_pos_index, :, one_batch_decision_dir_index,
                one_batch_dvalency_index, :] = predicted_decision_param.detach().numpy().reshape(one_batch_size, 1,
                                                                                                 target_decision_num)
        if child_only:
            decision_counter = decision_counter + self.param_smoothing
            decision_sum = np.sum(decision_counter, axis=4, keepdims=True)
            decision_param = decision_counter / decision_sum
        decision_counter = decision_counter + self.param_smoothing
        decision_sum = np.sum(decision_counter, axis=4, keepdims=True)
        decision_param_compare = decision_counter / decision_sum
        decision_difference = decision_param_compare - decision_param
        if not self.child_only:
            print 'distance for decision in this iteration '+str(LA.norm(decision_difference))
        trans_counter = trans_counter + self.param_smoothing
        child_sum = np.sum(trans_counter, axis=(1, 3), keepdims=True)
        trans_param = trans_counter / child_sum
        #trans_difference = trans_param_compare - trans_param
        #print 'distance for trans in this iteration ' + str(LA.norm(trans_difference))
        return trans_param, decision_param
