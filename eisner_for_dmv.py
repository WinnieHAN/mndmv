import numpy as np

import utils
from scipy.special import logsumexp


def batch_parse(batch_scores, batch_decision_score, valency_num, cvalency_num):
    batch_size, sentence_length, _, tag_num, _, _ = batch_scores.shape
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
                     batch_scores[:, r, l, :, :, :].swapaxes(1, 2).reshape(batch_size, 1, tag_num, tag_num,
                                                                           cvalency_num) \
                     + batch_decision_score[:, r, :, dir, :, 1].reshape(batch_size, 1, 1, tag_num, valency_num)
        else:
            span_i = ik_ci[:, :, :, :, 1].reshape(batch_size, num_ki, tag_num, 1, 1) \
                     + kj_ci[:, :, :, :, 0].reshape(batch_size, num_ki, 1, tag_num, 1) + \
                     batch_scores[:, l, r, :, :, :].reshape(batch_size, 1, tag_num, tag_num, cvalency_num) \
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


def batch_inside(batch_scores, batch_decision_score, valency_num, cvalency_num):
    batch_size, sentence_length, _, tag_num, _, _ = batch_scores.shape
    inside_complete_table = np.zeros((batch_size, sentence_length * sentence_length * 2, tag_num, valency_num))
    inside_incomplete_table = np.zeros(
        (batch_size, sentence_length * sentence_length * 2, tag_num, tag_num, valency_num))
    span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(sentence_length,
                                                                                             False)
    inside_complete_table.fill(-np.inf)
    inside_incomplete_table.fill(-np.inf)

    for ii in basic_span:
        (i, i, dir) = id_2_span[ii]
        inside_complete_table[:, ii, :, :] = batch_decision_score[:, i, :, dir, :, 0]

    for ij in ijss:
        (l, r, dir) = id_2_span[ij]
        # two complete span to form an incomplete span
        num_ki = len(ikis[ij])
        inside_ik_ci = inside_complete_table[:, ikis[ij], :, :].reshape(batch_size, num_ki, tag_num, 1, valency_num)
        inside_kj_ci = inside_complete_table[:, kjis[ij], :, :].reshape(batch_size, num_ki, 1, tag_num, valency_num)
        if dir == 0:
            span_inside_i = inside_ik_ci[:, :, :, :, 0].reshape(batch_size, num_ki, tag_num, 1, 1) \
                            + inside_kj_ci[:, :, :, :, 1].reshape(batch_size, num_ki, 1, tag_num, 1) \
                            + batch_scores[:, r, l, :, :, :].swapaxes(2, 1).reshape(batch_size, 1, tag_num, tag_num,
                                                                                    cvalency_num) \
                            + batch_decision_score[:, r, :, dir, :, 1].reshape(batch_size, 1, 1, tag_num, valency_num)

            # swap head-child to left-right position
        else:
            span_inside_i = inside_ik_ci[:, :, :, :, 1].reshape(batch_size, num_ki, tag_num, 1, 1) \
                            + inside_kj_ci[:, :, :, :, 0].reshape(batch_size, num_ki, 1, tag_num, 1) \
                            + batch_scores[:, l, r, :, :, :].reshape(batch_size, 1, tag_num, tag_num, cvalency_num) \
                            + batch_decision_score[:, l, :, dir, :, 1].reshape(batch_size, 1, tag_num, 1, valency_num)

        inside_incomplete_table[:, ij, :, :, :] = logsumexp(span_inside_i, axis=1)

        # one complete span and one incomplete span to form bigger complete span
        num_kc = len(ikcs[ij])
        if dir == 0:
            inside_ik_cc = inside_complete_table[:, ikcs[ij], :, :].reshape(batch_size, num_kc, tag_num, 1, valency_num)
            inside_kj_ic = inside_incomplete_table[:, kjcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num,
                                                                                 valency_num)
            span_inside_c = inside_ik_cc[:, :, :, :, 0].reshape(batch_size, num_kc, tag_num, 1, 1) + inside_kj_ic
            span_inside_c = span_inside_c.reshape(batch_size, num_kc * tag_num, tag_num, valency_num)
            inside_complete_table[:, ij, :, :] = logsumexp(span_inside_c, axis=1)
        else:
            inside_ik_ic = inside_incomplete_table[:, ikcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num,
                                                                                 valency_num)
            inside_kj_cc = inside_complete_table[:, kjcs[ij], :, :].reshape(batch_size, num_kc, 1, tag_num, valency_num)
            span_inside_c = inside_ik_ic + inside_kj_cc[:, :, :, :, 0].reshape(batch_size, num_kc, 1, tag_num, 1)
            span_inside_c = span_inside_c.swapaxes(3, 2).reshape(batch_size, num_kc * tag_num, tag_num, valency_num)
            # swap the left-right position since the left tags are to be indexed
            inside_complete_table[:, ij, :, :] = logsumexp(span_inside_c, axis=1)

    final_id = span_2_id[(0, sentence_length - 1, 1)]
    partition_score = inside_complete_table[:, final_id, 0, 0]

    return inside_complete_table, inside_incomplete_table, partition_score


def batch_outside(inside_complete_table, inside_incomplete_table, batch_scores, batch_decision_scores, valency_num,
                  cvalency_num):
    batch_size, sentence_length, _, tag_num, _, _ = batch_scores.shape
    outside_complete_table = np.zeros((batch_size, sentence_length * sentence_length * 2, tag_num, valency_num))
    outside_incomplete_table = np.zeros(
        (batch_size, sentence_length * sentence_length * 2, tag_num, tag_num, valency_num))
    span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(sentence_length, False)
    outside_complete_table.fill(-np.inf)
    outside_incomplete_table.fill(-np.inf)

    root_id = span_2_id.get((0, sentence_length - 1, 1))
    outside_complete_table[:, root_id, 0, 0] = 0.0

    complete_span_used_0 = set()
    complete_span_used_1 = set()
    incomplete_span_used = set()
    complete_span_used_0.add(root_id)

    for ij in reversed(ijss):
        (l, r, dir) = id_2_span[ij]
        # complete span consists of one incomplete span and one complete span
        num_kc = len(ikcs[ij])
        if dir == 0:
            outside_ij_cc = outside_complete_table[:, ij, :, :].reshape(batch_size, 1, 1, tag_num, valency_num)
            inside_kj_ic = inside_incomplete_table[:, kjcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num,
                                                                                 valency_num)
            inside_ik_cc = inside_complete_table[:, ikcs[ij], :, :].reshape(batch_size, num_kc, tag_num, 1, valency_num)
            outside_ik_cc = (outside_ij_cc + inside_kj_ic).swapaxes(2, 3)
            # swap left-right position since right tags are to be indexed
            outside_kj_ic = outside_ij_cc + inside_ik_cc[:, :, :, :, 0].reshape(batch_size, num_kc, tag_num, 1, 1)
            for i in range(num_kc):
                ik = ikcs[ij][i]
                kj = kjcs[ij][i]
                outside_ik_cc_i = logsumexp(outside_ik_cc[:, i, :, :, :], axis=(1, 3))
                if ik in complete_span_used_0:
                    outside_complete_table[:, ik, :, 0] = np.logaddexp(
                        outside_complete_table[:, ik, :, 0], outside_ik_cc_i)
                else:
                    outside_complete_table[:, ik, :, 0] = np.copy(outside_ik_cc_i)
                    complete_span_used_0.add(ik)

                if kj in incomplete_span_used:
                    outside_incomplete_table[:, kj, :, :, :] = np.logaddexp(outside_incomplete_table[:, kj, :, :, :],
                                                                            outside_kj_ic[:, i, :, :, :])
                else:
                    outside_incomplete_table[:, kj, :, :, :] = np.copy(outside_kj_ic[:, i, :, :, :])
                    incomplete_span_used.add(kj)
        else:
            outside_ij_cc = outside_complete_table[:, ij, :, :].reshape(batch_size, 1, tag_num, 1, valency_num)
            inside_ik_ic = inside_incomplete_table[:, ikcs[ij], :, :, :].reshape(batch_size, num_kc, tag_num, tag_num,
                                                                                 valency_num)
            inside_kj_cc = inside_complete_table[:, kjcs[ij], :, :].reshape(batch_size, num_kc, 1, tag_num, valency_num)
            outside_kj_cc = outside_ij_cc + inside_ik_ic
            outside_ik_ic = outside_ij_cc + inside_kj_cc[:, :, :, :, 0].reshape(batch_size, num_kc, 1, tag_num, 1)
            for i in range(num_kc):
                kj = kjcs[ij][i]
                ik = ikcs[ij][i]
                outside_kj_cc_i = logsumexp(outside_kj_cc[:, i, :, :, :], axis=(1, 3))
                if kj in complete_span_used_0:
                    outside_complete_table[:, kj, :, 0] = np.logaddexp(outside_complete_table[:, kj, :, 0],
                                                                       outside_kj_cc_i)
                else:
                    outside_complete_table[:, kj, :, 0] = np.copy(outside_kj_cc_i)
                    complete_span_used_0.add(kj)

                if ik in incomplete_span_used:
                    outside_incomplete_table[:, ik, :, :, :] = np.logaddexp(outside_incomplete_table[:, ik, :, :, :],
                                                                            outside_ik_ic[:, i, :, :, :])
                else:
                    outside_incomplete_table[:, ik, :, :, :] = np.copy(outside_ik_ic[:, i, :, :, :])
                    incomplete_span_used.add(ik)

        # incomplete span consists of two complete spans
        num_ki = len(ikis[ij])

        outside_ij_ii = outside_incomplete_table[:, ij, :, :, :].reshape(batch_size, 1, tag_num, tag_num, valency_num)
        inside_ik_ci = inside_complete_table[:, ikis[ij], :, :].reshape(batch_size, num_ki, tag_num, 1, valency_num)
        inside_kj_ci = inside_complete_table[:, kjis[ij], :].reshape(batch_size, num_ki, 1, tag_num, valency_num)

        if dir == 0:
            outside_ik_ci_0 = outside_ij_ii + inside_kj_ci[:, :, :, :, 1].reshape(batch_size, num_ki, 1, tag_num, 1) + \
                              batch_scores[:, r, l, :, :, :].swapaxes(1, 2). \
                                  reshape(batch_size, 1, tag_num, tag_num, cvalency_num) + \
                              batch_decision_scores[:, r, :, dir, :, 1].reshape(batch_size, 1, 1, tag_num, valency_num)

            outside_kj_ci_1 = outside_ij_ii + inside_ik_ci[:, :, :, :, 0].reshape(batch_size, num_ki, tag_num, 1, 1) + \
                              batch_scores[:, r, l, :, :, :].swapaxes(1, 2). \
                                  reshape(batch_size, 1, tag_num, tag_num, cvalency_num) + \
                              batch_decision_scores[:, r, :, dir, :, 1].reshape(batch_size, 1, 1, tag_num, valency_num)
        else:
            outside_ik_ci_1 = outside_ij_ii + inside_kj_ci[:, :, :, :, 0].reshape(batch_size, num_ki, 1, tag_num, 1) \
                              + batch_scores[:, l, r, :, :, :].reshape(batch_size, 1, tag_num, tag_num, cvalency_num) + \
                              batch_decision_scores[:, l, :, dir, :, 1].reshape(batch_size, 1, tag_num, 1, valency_num)
            outside_kj_ci_0 = outside_ij_ii + inside_ik_ci[:, :, :, :, 1].reshape(batch_size, num_ki, tag_num, 1, 1) + \
                              batch_scores[:, l, r, :, :, :].reshape(batch_size, 1, tag_num, tag_num, cvalency_num) + \
                              batch_decision_scores[:, l, :, dir, :, 1].reshape(batch_size, 1, tag_num, 1, valency_num)

        for i in range(num_ki):
            ik = ikis[ij][i]
            kj = kjis[ij][i]
            if dir == 0:
                outside_ik_ci_i_0 = logsumexp(outside_ik_ci_0[:, i, :, :, :], axis=(2, 3))
                outside_kj_ci_i_1 = logsumexp(outside_kj_ci_1[:, i, :, :, :], axis=(1, 3))
            else:
                outside_ik_ci_i_1 = logsumexp(outside_ik_ci_1[:, i, :, :, :], axis=(2, 3))
                outside_kj_ci_i_0 = logsumexp(outside_kj_ci_0[:, i, :, :, :], axis=(1, 3))
            if dir == 0:
                if ik in complete_span_used_0:
                    outside_complete_table[:, ik, :, 0] = np.logaddexp(outside_complete_table[:, ik, :, 0],
                                                                       outside_ik_ci_i_0)
                else:
                    outside_complete_table[:, ik, :, 0] = np.copy(outside_ik_ci_i_0)
                    complete_span_used_0.add(ik)

                if kj in complete_span_used_1:
                    outside_complete_table[:, kj, :, 1] = np.logaddexp(outside_complete_table[:, kj, :, 1],
                                                                       outside_kj_ci_i_1)
                else:
                    outside_complete_table[:, kj, :, 1] = outside_kj_ci_i_1
                    complete_span_used_1.add(kj)

            else:
                if ik in complete_span_used_1:
                    outside_complete_table[:, ik, :, 1] = np.logaddexp(outside_complete_table[:, ik, :, 1],
                                                                       outside_ik_ci_i_1)
                else:
                    outside_complete_table[:, ik, :, 1] = np.copy(outside_ik_ci_i_1)
                    complete_span_used_1.add(ik)

                if kj in complete_span_used_0:
                    outside_complete_table[:, kj, :, 0] = np.logaddexp(outside_complete_table[:, kj, :, 0],
                                                                       outside_kj_ci_i_0)
                else:
                    outside_complete_table[:, kj, :, 0] = outside_kj_ci_i_0
                    complete_span_used_0.add(kj)

    return outside_complete_table, outside_incomplete_table
