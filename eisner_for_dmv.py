import utils
import numpy as np


def batch_parse(batch_scores, batch_decision_score, valency_num):
    batch_size, sentence_length, _, tag_num, _ = batch_scores.shape
    # CYK table
    complete_table = np.zeros((batch_size, sentence_length * sentence_length * 2, tag_num, valency_num))
    incomplete_table = np.zeros((batch_size, sentence_length * sentence_length * 2, tag_num, tag_num, valency_num))
    complete_table.fill(-np.inf)
    incomplete_table.fill(-np.inf)
    # backtrack table
    complete_backtrack = -np.ones((batch_size, sentence_length * sentence_length * 2, tag_num, valency_num), dtype=int)
    incomplete_backtrack = -np.ones((batch_size, sentence_length * sentence_length * 2, tag_num, tag_num, valency_num),
                                    dtype=int)
    # span index table, to avoid redundant iterations
    span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(sentence_length, False)
    # initial basic complete spans
    for ii in basic_span:
        (i, i, dir) = id_2_span[ii]
        complete_table[:, ii, :, :] = batch_decision_score[:, i, :, dir, :, 0]
    for ij in ijss:
        (l, r, dir) = id_2_span[ij]
        num_ki = len(ikis[ij])
        ik_ci = complete_table[:, ikis[ij], :, :].reshape(batch_size, num_ki, tag_num, 1, valency_num)
        kj_ci = complete_table[:, kjis[ij], :, :].reshape(batch_size, num_ki, 1, tag_num, valency_num)
        # construct incomplete spans
        if dir == 0:
            span_i = ik_ci[:, :, :, :, 0].reshape(batch_size, num_ki, tag_num, 1, 1) \
                     + kj_ci[:, :, :, :, 1].reshape(batch_size, num_ki, 1, tag_num, 1) + \
                     batch_scores[:, r, l, :, :].swapaxes(1, 2).reshape(batch_size, 1, tag_num, tag_num, 1) \
                     + batch_decision_score[:, r, :, dir, :, 1].reshape(batch_size, 1, 1, tag_num, valency_num)
        else:
            span_i = ik_ci[:, :, :, :, 1].reshape(batch_size, num_ki, tag_num, 1, 1) \
                     + kj_ci[:, :, :, :, 0].reshape(batch_size, num_ki, 1, tag_num, 1) + \
                     batch_scores[:, l, r, :, :].reshape(batch_size, 1, tag_num, tag_num, 1) \
                     + batch_decision_score[:, l, :, dir, :, 1].reshape(batch_size, 1, tag_num, 1, valency_num)
        max = np.max(span_i, axis=1)

        incomplete_table[:, ij, :, :, :] = np.max(span_i, axis=1)
        max_idx = np.argmax(span_i, axis=1)
        incomplete_backtrack[:, ij, :, :, :] = max_idx
        # construct complete spans
        num_kc = len(ikcs[ij])
        if dir == 0:
            ik_cc = complete_table[:, ikcs[ij], :, :].reshape(batch_size, num_kc, tag_num, 1, valency_num)
            kj_ic = incomplete_table[:, kjcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num, valency_num)
            span_c = ik_cc[:, :, :, :, 0].reshape(batch_size, num_kc, tag_num, 1, 1) + kj_ic
            span_c = span_c.reshape(batch_size, num_kc * tag_num, tag_num, valency_num)
        else:
            ik_ic = incomplete_table[:, ikcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num, valency_num)
            kj_cc = complete_table[:, kjcs[ij], :, :].reshape(batch_size, num_kc, 1, tag_num, valency_num)
            span_c = ik_ic + kj_cc[:, :, :, :, 0].reshape(batch_size, num_kc, 1, tag_num, 1)
            span_c = span_c.swapaxes(2, 3).reshape(batch_size, num_kc * tag_num, tag_num, valency_num)
        complete_table[:, ij, :, :] = np.max(span_c, axis=1)
        max_idx = np.argmax(span_c, axis=1)
        complete_backtrack[:, ij, :, :] = max_idx

    tags = np.zeros((batch_size, sentence_length)).astype(int)
    heads = -np.ones((batch_size, sentence_length))
    head_valences = np.zeros((batch_size, sentence_length))
    valences = np.zeros((batch_size, sentence_length, 2))
    root_id = span_2_id[(0, sentence_length - 1, 1)]
    for s in range(batch_size):
        batch_backtracking(incomplete_backtrack, complete_backtrack, root_id, 0, 0, 0, 1, tags, heads, head_valences,
                           valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, s)

    return (heads, tags, head_valences, valences)


def batch_backtracking(incomplete_backtrack, complete_backtrack, span_id, l_tag, r_tag, decision_valence, complete,
                       tags, heads, head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num,
                       sen_id):

    (l, r, dir) = id_2_span[span_id]
    if l == r:
        valences[sen_id, l, dir] = decision_valence
        return
    if complete:
        if dir == 0:
            k = complete_backtrack[sen_id, span_id, r_tag, decision_valence]
            # print 'k is ', k, ' complete left'
            k_span, k_tag = utils.get_index(tag_num, k)
            left_span_id = ikcs[span_id][k_span]
            right_span_id = kjcs[span_id][k_span]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 0, k_tag, 0, 1, tags, heads,
                               head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, k_tag, r_tag, decision_valence,
                               0, tags, heads, head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id,
                               tag_num, sen_id)
            return
        else:
            num_k = len(ikcs[span_id])
            k = complete_backtrack[sen_id, span_id, l_tag, decision_valence]
            # print 'k is ', k, ' complete right'
            k_span, k_tag = utils.get_index(tag_num, k)
            left_span_id = ikcs[span_id][k_span]
            right_span_id = kjcs[span_id][k_span]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, l_tag, k_tag, decision_valence,
                               0, tags, heads, head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id,
                               tag_num, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, k_tag, 0, 0, 1, tags, heads,
                               head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            return
    else:
        if dir == 0:

            k = incomplete_backtrack[sen_id, span_id, l_tag, r_tag, decision_valence]
            # print 'k is ', k, ' incomplete left'
            heads[sen_id, l] = r
            tags[sen_id, l] = l_tag
            head_valences[sen_id, l] = decision_valence
            left_span_id = ikis[span_id][k]
            right_span_id = kjis[span_id][k]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, l_tag, 0, 0, 1, tags, heads,
                               head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 0, r_tag, 1, 1, tags, heads,
                               head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            return
        else:
            k = incomplete_backtrack[sen_id, span_id, l_tag, r_tag, decision_valence]
            # print 'k is', k, ' incomplete right'
            heads[sen_id, r] = l
            tags[sen_id, r] = r_tag
            head_valences[sen_id, r] = decision_valence
            left_span_id = ikis[span_id][k]
            right_span_id = kjis[span_id][k]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, l_tag, 0, 1, 1, tags, heads,
                               head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 0, r_tag, 0, 1, tags, heads,
                               head_valences, valences, ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            return
