import numpy as np
import itertools


import logging
logger = logging.getLogger("Test")

def get_beam_kept_gks(sim_matrix, pruning_rate = 0.5, dim = None, eval_kept_kernel_number = 1, inner_outer_balancer = 1, pruning_strategy = 'smooth_cost_beam', smoothness_list = None, beam_width = 3, smoothness_check_step = 2, cost_smooth_balancer = 0.5, show_analysis = False, eval_outer_cost_during_beam_search = True):

    def evaluate_selection_smoothness(selection_list, smoothness_list = smoothness_list):
        kept_kernel_smoothness = sum([smoothness_list[i][1] for i in selection_list])

        analysis_info = f'Selected kernels smoothness: {kept_kernel_smoothness}. Kernels <{selection_list}> evaluated (smoothness: {[smoothness_list[i][1] for i in selection_list]}).'

        # print('analysis_info: ', analysis_info)

        return kept_kernel_smoothness, analysis_info

    def evaluate_selection_cost(selection_list, eval_kept_kernel_number = eval_kept_kernel_number, eval_outer_cost = True):
        selected_kernel_pairs = list(itertools.combinations(selection_list, 2))
        selected_cost = 0
        for i in selected_kernel_pairs:
            # print(f'sim_matrix.shape: {sim_matrix.shape} ({i[0]}, {i[1]})')
            selected_cost += sim_matrix[i[0]][i[1]]
        cost = selected_cost
        analysis_info = f'{cost:.3f} = {selected_cost:.3f} kept ({len(selection_list)} kept kernels ({len(selected_kernel_pairs)} pairs)'

        if not eval_outer_cost or eval_kept_kernel_number == 0:
            return cost, analysis_info

        if eval_outer_cost and eval_kept_kernel_number != 0:
            pruned_cost = 0
            pruned_kernels = list(set([i for i in range(dim)]) - set(selection_list))
            # nonlocal eval_kept_kernel_ratio
            # eval_kept_kernel_number = len(selection_list) if eval_kept_kernel_ratio == 'all' else int(eval_kept_kernel_ratio * len(selection_list))

            # print(f'eval_kept_kernel_number {eval_kept_kernel_number} = eval_kept_kernel_ratio {eval_kept_kernel_ratio} * len(selection_list) {len(selection_list)}')
            # pruned_kernel_pairs = list(itertools.product(pruned_kernels, selection_list))
            # for i in pruned_kernel_pairs:
            #     pruned_cost += sim_matrix[i[0]][i[1]]
            for i in pruned_kernels:
                # min_dis_pair_cost = max([sim_matrix[j][i] for j in selection_list])

                if eval_kept_kernel_number == 1:
                    min_dis_pairs_cost = [max([sim_matrix[j][i] for j in selection_list])]
                else:
                    min_dis_pairs_cost = sorted([sim_matrix[j][i] for j in selection_list], reverse = True)[:eval_kept_kernel_number]
                pruned_cost += sum(min_dis_pairs_cost)

            nonlocal inner_outer_balancer
            if inner_outer_balancer == 'auto':
                try:
                    inner_outer_balancer = (len(selected_kernel_pairs) / (len(pruned_kernels) *  eval_kept_kernel_number))
                except ZeroDivisionError:
                    inner_outer_balancer = 0

            cost = selected_cost - (pruned_cost * inner_outer_balancer)
            analysis_info = f'{cost:.3f} = {selected_cost:.3f} kept - {inner_outer_balancer:.3f} * {pruned_cost:.3f} pruned [{len(selection_list)} kept kernels ({len(selected_kernel_pairs)} pairs) & {eval_kept_kernel_number}-{len(pruned_kernels)} kept-pruned kernal pairs evaluated ({len(pruned_kernels) *  eval_kept_kernel_number} pairs)]'


        return cost, analysis_info




    def get_relative_cost(selected_gks, target_gk):
        cost = 0
        for selected_gk in selected_gks:
            cost += sim_matrix[selected_gk][target_gk]
        return cost

    def get_next_beam(current_beam, last_gk_flag = False):


        next_beam = []
        # print(f'current_beam: {current_beam}')

        current_beam_encounted_selected_gks = {}
        # print('current_beam', current_beam)
        for current_gk_list in current_beam:
            next_gk_candidate_list = [i for i in range(dim) if i not in current_gk_list]


            next_gk_candidate_list = [(next_gk_index, frozenset(current_gk_list + (next_gk_index,))) for next_gk_index in next_gk_candidate_list]
            next_gk_candidate_list = [*{x[-1]: x[0] for x in next_gk_candidate_list}.values()]

            next_gk_cost_list = [(next_gk_index, current_gk_list, get_relative_cost(current_gk_list, next_gk_index)) for next_gk_index in next_gk_candidate_list]
            next_gk_cost_list = sorted(next_gk_cost_list, key = lambda x:x[-1])[:beam_width]


            beam_candidate = [current_gk_list + (next_gk_index,) for next_gk_index, current_gk_list, cost in next_gk_cost_list]
            next_beam.extend(beam_candidate)


        return next_beam

    def normalize_lists(list1, list2, new_min=0, new_max=1):
        combined_min = min(min(list1), min(list2))
        combined_max = max(max(list1), max(list2))

        def normalize(value, old_min, old_max, new_min, new_max):
            return ((value - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

        normalized_list1 = [normalize(value, combined_min, combined_max, new_min, new_max) for value in list1]
        normalized_list2 = [normalize(value, combined_min, combined_max, new_min, new_max) for value in list2]

        return normalized_list1, normalized_list2

    dim = sim_matrix.shape[0] if dim is None else dim
    remained_kernel_capacity = int((1 - pruning_rate) * (dim + 1))

    # print(f'remained_kernel_capacity: {remained_kernel_capacity}')

    all_selection_list = []
    for initial_kernel_index in range(dim):



        next_beam = [(initial_kernel_index,)]
        current_gk_amount = len(next_beam[0])

        beam_step_counter = 0
        while current_gk_amount < remained_kernel_capacity:

            next_beam = get_next_beam(next_beam)
            current_gk_amount = len(next_beam[0])
            beam_step_counter += 1

            if beam_step_counter % smoothness_check_step == 0:
                if eval_outer_cost_during_beam_search:
                    next_beam = [(candidate_gk_list, evaluate_selection_cost(candidate_gk_list, eval_kept_kernel_number, eval_outer_cost = True)[0], evaluate_selection_smoothness(candidate_gk_list)[0]) for candidate_gk_list in next_beam]
                else:
                    next_beam = [(candidate_gk_list, evaluate_selection_cost(candidate_gk_list, eval_kept_kernel_number, eval_outer_cost = False)[0], evaluate_selection_smoothness(candidate_gk_list)[0]) for candidate_gk_list in next_beam]


                ################################################################

                # # # Normal Rank

                # next_beam = [(candidate_gk_list, candidate_gk_cost_rank, candidate_gk_smoothness) for candidate_gk_cost_rank, (candidate_gk_list, candidate_gk_cost, candidate_gk_smoothness) in enumerate(sorted(next_beam, key = lambda x:x[-2]))]
                # next_beam = [(candidate_gk_list, candidate_gk_cost_rank, candidate_gk_smoothness_rank) for candidate_gk_smoothness_rank, (candidate_gk_list, candidate_gk_cost_rank, candidate_gk_smoothness) in enumerate(sorted(next_beam, key = lambda x:x[-1]))]
                # next_beam = [(candidate_gk_list, cost_smooth_balancer * candidate_gk_cost_rank + (1 - cost_smooth_balancer) * candidate_gk_smoothness_rank) for candidate_gk_list, candidate_gk_cost_rank, candidate_gk_smoothness_rank in next_beam]
                # next_beam = [gk_list for gk_list, gk_cost_smoothness_rank_sum in sorted(next_beam, key = lambda x:x[-1])[:beam_width]]

                # # # a * c + (1 - a) s
                # # # = a * c + s - as
                # # # = s + a(c - s)

                # # Reversed Rank
                next_beam = [(candidate_gk_list, candidate_gk_cost_rank, candidate_gk_smoothness) for candidate_gk_cost_rank, (candidate_gk_list, candidate_gk_cost, candidate_gk_smoothness) in enumerate(sorted(next_beam, key = lambda x:x[-2], reverse = True))]
                next_beam = [(candidate_gk_list, candidate_gk_cost_rank, candidate_gk_smoothness_rank) for candidate_gk_smoothness_rank, (candidate_gk_list, candidate_gk_cost_rank, candidate_gk_smoothness) in enumerate(sorted(next_beam, key = lambda x:x[-1], reverse = True))]
                next_beam = [(candidate_gk_list, cost_smooth_balancer * candidate_gk_cost_rank + (1 - cost_smooth_balancer) * candidate_gk_smoothness_rank) for candidate_gk_list, candidate_gk_cost_rank, candidate_gk_smoothness_rank in next_beam]
                next_beam = [gk_list for gk_list, gk_cost_smoothness_rank_sum in sorted(next_beam, key = lambda x:x[-1], reverse = True)[:beam_width]]

                # Normalized Add
                # total_cost_list = [total_cost for candidate_gk_list, total_cost, total_smoothness in next_beam]
                # total_smoothness_list = [total_smoothness for candidate_gk_list, total_cost, total_smoothness in next_beam]
                # normalized_total_cost_list, normalized_total_smoothness_list = normalize_lists(total_cost_list, total_smoothness_list)
                # next_beam = [(next_beam_output[0], cost_smooth_balancer * normalized_total_cost + (1 - cost_smooth_balancer) * normalized_total_smoothness) for next_beam_output, normalized_total_cost, normalized_total_smoothness in zip(next_beam, total_cost_list, total_smoothness_list)]
                # next_beam = [gk_list for gk_list, gk_cost_smoothness_normalized_score_sum in sorted(next_beam, key = lambda x:x[-1])[:beam_width]]

                ################################################################



        all_selection_list.extend(next_beam)

    selection_list_cost_arena = []
    for i, candidate_selection_list in enumerate(all_selection_list):
        if pruning_strategy == 'smooth_cost_beam':
            kept_kernels_cost, analysis_info = evaluate_selection_cost(candidate_selection_list, eval_outer_cost = True)
        elif pruning_strategy == 'smooth_cost_beam_smooth_select':
            kept_kernels_cost, analysis_info = evaluate_selection_smoothness(candidate_selection_list, smoothness_list)

        selection_list_cost_arena.append((i, kept_kernels_cost, analysis_info))

    min_selection_index, min_selection_cost, min_analysis_info = min(selection_list_cost_arena, key = lambda x:x[1])

    # print("#"*10)
    # print('final analysis: ', min_selection_index, min_selection_cost, min_analysis_info)
    # print("#"*10)


    if show_analysis:
        for i, j in zip(all_selection_list, [k for k in selection_list_cost_arena]):
            print(f'candidate list: {i}; cost: {j[1]} as {j[2]}')

        print(f'Cost: {min_analysis_info};\n\t {len(all_selection_list[min_selection_index])} kernels kept: {all_selection_list[min_selection_index]} (pruning rate {pruning_rate});')

    return all_selection_list[min_selection_index]