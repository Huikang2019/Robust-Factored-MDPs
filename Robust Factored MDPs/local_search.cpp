#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern double  prob_test[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern double  prob_hist[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern double  prob_real[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern int     basis[number_of_computers][scope];
extern double  rewards[number_of_states][number_of_computers];

extern int      state_constr[number_of_computers][number_of_constraints];
extern int      original_constr[number_of_computers][number_of_constraints];
extern int      action_constr[number_of_computers][number_of_constraints];
extern int      time_constr[number_of_constraints];

double state_constraint_value(NumArray3D& weight, NumArray3D& weight_beta, NumArray3D& weight_gamma, NumArray2DA& weight_eta, int* current_state,
    int* original_state, int* current_action, const int time) {
    double prob_i, prob_j;
    int i, j;
    double curr_LHS = 0, curr_RHS = 0;
    if (time == 0) {
        for (int k = 0; k < number_of_computers; k++)
            curr_LHS -= rewards[current_state[k]][k];
        for (int k = 0; k < number_of_computers; k++) {
            curr_LHS += weight[current_state[basis[k][0]]][current_state[basis[k][1]]][k]
                - weight_gamma[original_state[basis[k][0]]][original_state[basis[k][1]]][k]
                - weight_eta[current_action[k]][k];
        }
        /*for (int k = 0; k < number_of_computers; k++) {
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    i = basis[k][0];
                    prob_i = prob_out(current_state, original_state, i, current_action[i], state_i, 0);
                    j = basis[k][1];
                    prob_j = prob_out(current_state, original_state, j, current_action[j], state_j, 0);
                    curr_RHS += prob_i * prob_j * weight_beta[state_i][state_j][k];
                }
            }
        }*/
        curr_RHS = trans_prob_update(-2, weight, weight_beta, current_state, current_action, original_state, time);
        return curr_LHS - curr_RHS;
    }
    else {
        for (int k = 0; k < number_of_computers; k++) {
            curr_LHS += weight_beta[current_state[basis[k][0]]][current_state[basis[k][1]]][k]
                + weight_gamma[original_state[basis[k][0]]][original_state[basis[k][1]]][k]
                + weight_eta[current_action[k]][k];
        }
        /*for (int k = 0; k < number_of_computers; k++) {
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    i = basis[k][0];
                    prob_i = prob_out(current_state, original_state, i, current_action[i], state_i, 1);
                    j = basis[k][1];
                    prob_j = prob_out(current_state, original_state, j, current_action[j], state_j, 1);
                    curr_RHS += prob_i * prob_j * weight[state_i][state_j][k];
                }
            }
        }*/
        curr_RHS = trans_prob_update(-2, weight, weight_beta, current_state, current_action, original_state, time);
        return curr_LHS - discount_factor * curr_RHS;
    }

}

void random_generate_constraints_local_search(NumArray3D& weight, NumArray3D& weight_beta, NumArray3D& weight_gamma, NumArray2DA& weight_eta,
    int number_of_constr, int& constr_num) {
    int current_state[number_of_computers], current_state_next[number_of_computers], action_next[number_of_computers],
        current_action[number_of_computers], original_state[number_of_computers], original_state_next[number_of_computers], time;

    double value_next, value_best;
    for (int number = 0; number < number_of_constr; number++) {
        generate_state_action_half(current_state, original_state, current_action, time);
        for (int i = 0; i < number_of_computers; i++) {
            current_state_next[i] = current_state[i];
            original_state_next[i] = original_state[i];
        }
        value_best = state_constraint_value(weight, weight_beta, weight_gamma, weight_eta, current_state, original_state, current_action, time);
        if (time == 0) {
            for (int i = 0; i < number_of_computers; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    if (j != current_state[i]) {
                        current_state_next[i] = j;
                        original_state_next[i] = j;
                        value_next = state_constraint_value(weight, weight_beta, weight_gamma, weight_eta, current_state_next,
                            original_state_next, current_action, time);
                        if (value_next < value_best) {
                            current_state[i] = j;
                            original_state[i] = j;
                            value_best = value_next;
                        }
                    }
                }
                current_state_next[i] = current_state[i];
                original_state_next[i] = original_state[i];
            }
        }
        else {
            for (int i = 0; i < number_of_computers; i++) {
                if (i % 2 == 0) {
                    if (current_action[i] == 1) {
                        current_state[i] = 2;
                        continue;
                    }
                    for (int j = 0; j < number_of_states; j++) {
                        for (int jj = 0; jj < number_of_states; jj++) {
                            current_state_next[i] = j;
                            original_state_next[i] = jj;
                            value_next = state_constraint_value(weight, weight_beta, weight_gamma, weight_eta, current_state_next,
                                original_state_next, current_action, time);
                            if (value_next < value_best) {
                                current_state[i] = j;
                                original_state[i] = jj;
                                value_best = value_next;
                            }
                        }
                    }
                    current_state_next[i] = current_state[i];
                    original_state_next[i] = original_state[i];
                }
                else {
                    for (int j = 0; j < number_of_states; j++) {
                        if (j != current_state[i]) {
                            current_state_next[i] = j;
                            original_state_next[i] = j;
                            value_next = state_constraint_value(weight, weight_beta, weight_gamma, weight_eta, current_state_next,
                                original_state_next, current_action, time);
                            if (value_next < value_best) {
                                current_state[i] = j;
                                original_state[i] = j;
                                value_best = value_next;
                            }
                        }
                    }
                    current_state_next[i] = current_state[i];
                    original_state_next[i] = original_state[i];
                }
            }
        }

        for (int i = 0; i < number_of_computers; ++i) {
            action_next[i] = 0;
        }

        /*for (int i = 0; i < number_of_computers; i++) {
            action_next[i] = 1;
            for (int k = i + 1; k < number_of_computers; k++) {
                action_next[k] = 1;
                value_next = state_constraint_value(weight, weight_beta, weight_gamma, weight_eta, current_state, original_state, action_next, time);
                if (value_next < value_best) {
                    for (int j = 0; j < number_of_computers; ++j) {
                        current_action[j] = action_next[j];
                    }
                    value_best = value_next;
                }
                action_next[k] = 0;
            }

            action_next[i] = 0;
        }*/

        for (int i = 0; i < number_of_computers; i++) {
            if ((time == 0) || ((time == 1) && (i % 2 == 1))) {
                if (current_state[i] != original_state[i]) cout << "error!!!" << endl;
            }
            state_constr[i][constr_num] = current_state[i];
            action_constr[i][constr_num] = current_action[i];
            original_constr[i][constr_num] = original_state[i];
        }
        time_constr[constr_num] = time;

        trans_prob_update(constr_num, weight, weight_beta, current_state, current_action, original_state, time);

        constr_num++;
        if (constr_num >= number_of_constraints) break;
    }

}

double compare_two_states(NumArray3D& weight, const int* current_state_1,
    const int* current_state_2, const int* current_action) {

    double difference = 0, prob1_i, prob1_j, prob2_i, prob2_j;
    int i, j;
    for (int i = 0; i < number_of_computers; ++i)
        difference -= rewards[current_state_1[i]][i] - rewards[current_state_2[i]][i];
    for (int k = 0; k < number_of_computers; k++) {
        i = basis[k][0];
        j = basis[k][1];
        if ((current_state_1[i] != current_state_2[i]) || (current_state_1[j] != current_state_2[j])
            || (current_state_1[neighbors[i][0]] != current_state_2[neighbors[i][0]])
            || (current_state_1[neighbors[j][0]] != current_state_2[neighbors[j][0]])
            || (current_state_1[neighbors[i][1]] != current_state_2[neighbors[i][1]])
            || (current_state_1[neighbors[j][1]] != current_state_2[neighbors[j][1]])) {
            difference += weight[current_state_1[i]][current_state_1[j]][k] - weight[current_state_2[i]][current_state_2[j]][k];
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    if (current_action[i] == 1) {
                        if (state_i == number_of_states - 1) { prob1_i = 1.0;  prob2_i = 1.0; }
                        else { prob1_i = 0.0;  prob2_i = 0.0; }
                    }
                    else {
                        prob1_i = prob_hist[state_i][current_state_1[i]][current_state_1[neighbors[i][0]]][current_state_1[neighbors[i][1]]][i];
                        prob2_i = prob_hist[state_i][current_state_2[i]][current_state_2[neighbors[i][0]]][current_state_2[neighbors[i][1]]][i];
                    }
                    if (current_action[j] == 1) {
                        if (state_j == number_of_states - 1) { prob1_j = 1.0;  prob2_j = 1.0; }
                        else { prob1_j = 0.0;  prob2_j = 0.0; }
                    }
                    else {
                        prob1_j = prob_hist[state_j][current_state_1[j]][current_state_1[neighbors[j][0]]][current_state_1[neighbors[j][1]]][j];
                        prob2_j = prob_hist[state_j][current_state_2[j]][current_state_2[neighbors[j][0]]][current_state_2[neighbors[j][1]]][j];
                    }
                    difference -= discount_factor * (prob1_i * prob1_j - prob2_i * prob2_j) * weight[state_i][state_j][k];
                }
            }
        }
    }

    return difference;
}

double compare_two_states_true(NumArray3D& weight, const int* current_state_1,
    const int* current_state_2, const int* current_action) {

    double difference = 0, prob1_i, prob1_j, prob2_i, prob2_j;
    int i, j;
    for (int i = 0; i < number_of_computers; ++i)
        difference -= rewards[current_state_1[i]][i] - rewards[current_state_2[i]][i];
    for (int k = 0; k < number_of_computers; k++) {
        i = basis[k][0];
        j = basis[k][1];
        if ((current_state_1[i] != current_state_2[i]) || (current_state_1[j] != current_state_2[j])
            || (current_state_1[neighbors[i][0]] != current_state_2[neighbors[i][0]])
            || (current_state_1[neighbors[j][0]] != current_state_2[neighbors[j][0]])
            || (current_state_1[neighbors[i][1]] != current_state_2[neighbors[i][1]])
            || (current_state_1[neighbors[j][1]] != current_state_2[neighbors[j][1]])) {
            difference += weight[current_state_1[i]][current_state_1[j]][k] - weight[current_state_2[i]][current_state_2[j]][k];
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    if (current_action[i] == 1) {
                        if (state_i == number_of_states - 1) { prob1_i = 1.0;  prob2_i = 1.0; }
                        else { prob1_i = 0.0;  prob2_i = 0.0; }
                    }
                    else {
                        prob1_i = prob_real[state_i][current_state_1[i]][current_state_1[neighbors[i][0]]][current_state_1[neighbors[i][1]]][i];
                        prob2_i = prob_real[state_i][current_state_2[i]][current_state_2[neighbors[i][0]]][current_state_2[neighbors[i][1]]][i];
                    }
                    if (current_action[j] == 1) {
                        if (state_j == number_of_states - 1) { prob1_j = 1.0;  prob2_j = 1.0; }
                        else { prob1_j = 0.0;  prob2_j = 0.0; }
                    }
                    else {
                        prob1_j = prob_real[state_j][current_state_1[j]][current_state_1[neighbors[j][0]]][current_state_1[neighbors[j][1]]][j];
                        prob2_j = prob_real[state_j][current_state_2[j]][current_state_2[neighbors[j][0]]][current_state_2[neighbors[j][1]]][j];
                    }
                    difference -= discount_factor * (prob1_i * prob1_j - prob2_i * prob2_j) * weight[state_i][state_j][k];
                }
            }
        }
    }

    return difference;
}

void random_generate_constraints_local_search_NonRobust(GRBModel& model, NumVar3D& alpha, NumArray3D& weight,
    int number_of_constr, int& constr_num) {
    int current_state[number_of_computers], current_state_next[number_of_computers], current_action[number_of_computers];

    for (int number = 0; number < number_of_constr; number++) {
        generate_state_action(current_state, current_action);
        for (int i = 0; i < number_of_computers; i++)
            current_state_next[i] = current_state[i];
        for (int i = 0; i < number_of_computers; i++) {
            for (int j = 0; j < number_of_states; j++) {
                if (j != current_state[i]) {
                    current_state_next[i] = j;
                    if (compare_two_states(weight, current_state, current_state_next, current_action) > 0) {
                        current_state[i] = j;
                    }
                }
            }
            current_state_next[i] = current_state[i];
        }
        add_constraint_to_master_problem(model, alpha, current_state, current_action, constr_num);
    }

}

void random_generate_constraints_local_search_true(GRBModel& model, NumVar3D& alpha, NumArray3D& weight,
    int number_of_constr) {
    int current_state[number_of_computers], current_state_next[number_of_computers], current_action[number_of_computers];

    for (int number = 0; number < number_of_constr; number++) {
        generate_state_action(current_state, current_action);
        for (int i = 0; i < number_of_computers; i++)
            current_state_next[i] = current_state[i];
        for (int i = 0; i < number_of_computers; i++) {
            for (int j = 0; j < number_of_states; j++) {
                if (j != current_state[i]) {
                    current_state_next[i] = j;
                    if (compare_two_states_true(weight, current_state, current_state_next, current_action) > 0) {
                        current_state[i] = j;
                    }
                }
            }
            current_state_next[i] = current_state[i];
        }
        add_constraint_to_master_problem_true(model, alpha, current_state, current_action);
    }

}