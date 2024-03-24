#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern int     basis[number_of_basis][scope];

double compare_two_states_scope_one(NumArray2D& weight, const int* current_state_1,
    const int* current_state_2, const int* current_action) {

    double difference = 0, prob1_i, prob2_i;
    int i;
    for (int i = 0; i < number_of_computers; ++i)
        difference -= (double)(current_state_1[i] - current_state_2[i]);
    for (int k = 0; k < number_of_basis; k++) {
        i = basis[k][0];
        if ((current_state_1[i] != current_state_2[i])
            || (current_state_1[neighbors[i][0]] != current_state_2[neighbors[i][0]])
            || (current_state_1[neighbors[i][1]] != current_state_2[neighbors[i][1]])) {
            difference += weight[current_state_1[i]][k] - weight[current_state_2[i]][k];
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                if (current_action[i] == 1) {
                    if (state_i == number_of_states - 1) { prob1_i = 1.0;  prob2_i = 1.0; }
                    else { prob1_i = 0.0;  prob2_i = 0.0; }
                }
                else {
                    prob1_i = prob[state_i][current_state_1[i]][current_state_1[neighbors[i][0]]][current_state_1[neighbors[i][1]]];
                    prob2_i = prob[state_i][current_state_2[i]][current_state_2[neighbors[i][0]]][current_state_2[neighbors[i][1]]];
                }
                difference -= discount_factor * (prob1_i - prob2_i) * weight[state_i][k];
            }
        }
    }

    return difference;
}

double compare_two_states_scope_two(NumArray3D& weight, const int* current_state_1,
    const int* current_state_2, const int* current_action) {

    double difference = 0, prob1_i, prob1_j, prob2_i, prob2_j;
    int i, j;
    for (int i = 0; i < number_of_computers; ++i)
        difference -= (double)(current_state_1[i] - current_state_2[i]);
    for (int k = 0; k < number_of_basis; k++) {
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
                        prob1_i = prob[state_i][current_state_1[i]][current_state_1[neighbors[i][0]]][current_state_1[neighbors[i][1]]];
                        prob2_i = prob[state_i][current_state_2[i]][current_state_2[neighbors[i][0]]][current_state_2[neighbors[i][1]]];
                    }
                    if (current_action[j] == 1) {
                        if (state_j == number_of_states - 1) { prob1_j = 1.0;  prob2_j = 1.0; }
                        else { prob1_j = 0.0;  prob2_j = 0.0; }
                    }
                    else {
                        prob1_j = prob[state_j][current_state_1[j]][current_state_1[neighbors[j][0]]][current_state_1[neighbors[j][1]]];
                        prob2_j = prob[state_j][current_state_2[j]][current_state_2[neighbors[j][0]]][current_state_2[neighbors[j][1]]];
                    }
                    difference -= discount_factor * (prob1_i * prob1_j - prob2_i * prob2_j) * weight[state_i][state_j][k];
                }
            }
        }
    }

    return difference;
}

double compare_two_states_scope_three(NumArray4D& weight, const int* current_state_1,
    const int* current_state_2, const int* current_action) {

    double difference = 0, prob1_i, prob1_j, prob1_l, prob2_i, prob2_j, prob2_l;
    int i, j, l;
    for (int i = 0; i < number_of_computers; ++i)
        difference -= (double)(current_state_1[i] - current_state_2[i]);
    for (int k = 0; k < number_of_basis; k++) {
        i = basis[k][0];
        j = basis[k][1];
        l = basis[k][2];
        if ((current_state_1[i] != current_state_2[i]) || (current_state_1[j] != current_state_2[j])
            || (current_state_1[l] != current_state_2[l])
            || (current_state_1[neighbors[i][0]] != current_state_2[neighbors[i][0]])
            || (current_state_1[neighbors[j][0]] != current_state_2[neighbors[j][0]])
            || (current_state_1[neighbors[l][0]] != current_state_2[neighbors[l][0]])
            || (current_state_1[neighbors[i][1]] != current_state_2[neighbors[i][1]])
            || (current_state_1[neighbors[j][1]] != current_state_2[neighbors[j][1]])
            || (current_state_1[neighbors[l][1]] != current_state_2[neighbors[l][1]])) {
            difference += weight[current_state_1[i]][current_state_1[j]][current_state_1[l]][k] -
                weight[current_state_2[i]][current_state_2[j]][current_state_2[l]][k];
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    for (int state_l = 0; state_l < number_of_states; state_l++) {
                        if (current_action[i] == 1) {
                            if (state_i == number_of_states - 1) { prob1_i = 1.0;  prob2_i = 1.0; }
                            else { prob1_i = 0.0;  prob2_i = 0.0; }
                        }
                        else {
                            prob1_i = prob[state_i][current_state_1[i]][current_state_1[neighbors[i][0]]][current_state_1[neighbors[i][1]]];
                            prob2_i = prob[state_i][current_state_2[i]][current_state_2[neighbors[i][0]]][current_state_2[neighbors[i][1]]];
                        }
                        if (current_action[j] == 1) {
                            if (state_j == number_of_states - 1) { prob1_j = 1.0;  prob2_j = 1.0; }
                            else { prob1_j = 0.0;  prob2_j = 0.0; }
                        }
                        else {
                            prob1_j = prob[state_j][current_state_1[j]][current_state_1[neighbors[j][0]]][current_state_1[neighbors[j][1]]];
                            prob2_j = prob[state_j][current_state_2[j]][current_state_2[neighbors[j][0]]][current_state_2[neighbors[j][1]]];
                        }
                        if (current_action[l] == 1) {
                            if (state_l == number_of_states - 1) { prob1_l = 1.0;  prob2_l = 1.0; }
                            else { prob1_l = 0.0;  prob2_l = 0.0; }
                        }
                        else {
                            prob1_l = prob[state_l][current_state_1[l]][current_state_1[neighbors[l][0]]][current_state_1[neighbors[l][1]]];
                            prob2_l = prob[state_l][current_state_2[l]][current_state_2[neighbors[l][0]]][current_state_2[neighbors[l][1]]];
                        }
                        difference -= discount_factor * (prob1_i * prob1_j * prob1_l - prob2_i * prob2_j * prob2_l) * weight[state_i][state_j][state_l][k];
                    }
                }
            }
        }
    }

    return difference;
}

double state_action_value_scope_three(NumArray4D& weight, const int* current_state, const int* current_action) {

    double curr_LHS = 0, curr_RHS = 0;
    for (int i = 0; i < number_of_computers; ++i)
        curr_LHS -= current_state[i];
    for (int k = 0; k < number_of_basis; k++) {
        curr_LHS += weight[current_state[basis[k][0]]][current_state[basis[k][1]]][current_state[basis[k][2]]][k];
    }

    double prob_i, prob_j, prob_l;
    int i, j, l;
    for (int state_i = 0; state_i < number_of_states; ++state_i) {
        for (int state_j = 0; state_j < number_of_states; state_j++) {
            for (int state_l = 0; state_l < number_of_states; ++state_l) {
                for (int k = 0; k < number_of_basis; k++) {
                    i = basis[k][0];
                    if (current_action[i] == 1) {
                        if (state_i == number_of_states - 1) prob_i = 1.0;
                        else prob_i = 0.0;
                    }
                    else {
                        prob_i = prob[state_i][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]];
                    }
                    j = basis[k][1];
                    if (current_action[j] == 1) {
                        if (state_j == number_of_states - 1) prob_j = 1.0;
                        else prob_j = 0.0;
                    }
                    else {
                        prob_j = prob[state_j][current_state[j]][current_state[neighbors[j][0]]][current_state[neighbors[j][1]]];
                    }
                    l = basis[k][2];
                    if (current_action[l] == 1) {
                        if (state_l == number_of_states - 1) prob_l = 1.0;
                        else prob_l = 0.0;
                    }
                    else {
                        prob_l = prob[state_l][current_state[l]][current_state[neighbors[l][0]]][current_state[neighbors[l][1]]];
                    }
                    curr_RHS += prob_i * prob_j * prob_l * weight[state_i][state_j][state_l][k];
                }
            }
        }
    }
    return curr_LHS - discount_factor * curr_RHS;
}

void random_generate_constraints_local_search_scope_one(GRBModel& model, NumVar2D& alpha, NumArray2D& weight, int inner_iter) {
    int current_state[number_of_computers], current_state_next[number_of_computers], current_action[number_of_computers];

    for (int number = inner_iter + 1; number < number_of_computers; number++) {
        generate_state_action(current_state, current_action);
        for (int i = 0; i < number_of_computers; i++) {
            current_state_next[i] = current_state[i];
            //current_action[i] = 0;
        }
        //current_action[inner_iter] = 1;
        //current_action[number] = 1;

        for (int i = 0; i < number_of_computers; i++) {
            for (int j = 0; j < number_of_states; j++) {
                if (j != current_state[i]) {
                    current_state_next[i] = j;
                    if (compare_two_states_scope_one(weight, current_state, current_state_next, current_action) > 0) {
                        current_state[i] = j;
                    }
                }
            }
            current_state_next[i] = current_state[i];
        }
        add_constraint_to_master_problem_scope_one(model, alpha, current_state, current_action);
    }

}

void random_generate_constraints_local_search_scope_two(GRBModel& model, NumVar3D& alpha, NumArray3D& weight, int inner_iter) {
    int current_state[number_of_computers], current_state_next[number_of_computers], current_action[number_of_computers];

    for (int number = inner_iter + 1; number < number_of_computers; number++) {
        generate_state_action(current_state, current_action);
        for (int i = 0; i < number_of_computers; i++) {
            current_state_next[i] = current_state[i];
        }

        for (int i = 0; i < number_of_computers; i++) {
            for (int j = 0; j < number_of_states; j++) {
                if (j != current_state[i]) {
                    current_state_next[i] = j;
                    if (compare_two_states_scope_two(weight, current_state, current_state_next, current_action) > 0) {
                        current_state[i] = j;
                    }
                }
            }
            current_state_next[i] = current_state[i];
        }
        add_constraint_to_master_problem_scope_two(model, alpha, current_state, current_action);
    }

}

void random_generate_constraints_local_search_scope_three(GRBModel& model, NumVar4D& alpha, NumArray4D& weight, int inner_iter) {
    int current_state[number_of_computers], current_state_next[number_of_computers], current_action[number_of_computers];

    for (int number = inner_iter + 1; number < number_of_computers; number++) {
        generate_state_action(current_state, current_action);
        //for (int i = 0; i < number_of_computers; i++)
            //current_action[i] = 0;
        //current_action[number] = 1;
        //current_action[inner_iter] = 1;
        for (int i = 0; i < number_of_computers; i++)
            current_state_next[i] = current_state[i];
        for (int i = 0; i < number_of_computers; i++) {
            for (int j = 0; j < number_of_states; j++) {
                if (j != current_state[i]) {
                    current_state_next[i] = j;
                    if (compare_two_states_scope_three(weight, current_state, current_state_next, current_action) > 0) {                   
                        current_state[i] = j;
                    }
                }
            }
            current_state_next[i] = current_state[i];
        }
        add_constraint_to_master_problem_scope_three(model, alpha, current_state, current_action);
    }

}

void fixed_constraints_local_search_scope_one(GRBModel& model, NumVar2D& alpha, NumArray2D& weight, int* state, int* action) {
    int current_state[number_of_computers], current_state_next[number_of_computers], current_action[number_of_computers];
    int state0, action0;
    for (int number = 0; number < number_of_computers; number++) {

        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = state[i];
            current_action[i] = action[i];
            current_state_next[i] = current_state[i];
        }

        for (int i = 0; i < number_of_computers; i++) {
            for (int j = 0; j < number_of_states; j++) {
                if (j != current_state[i]) {
                    current_state_next[i] = j;
                    if (compare_two_states_scope_one(weight, current_state, current_state_next, current_action) > 0) {
                        //current_state[i] = j;
                    }
                }
            }
            current_state_next[i] = current_state[i];
        }
        add_constraint_to_master_problem_scope_one(model, alpha, current_state, current_action);

        state0 = state[0], action0 = action[0];
        for (int j = 0; j < number_of_computers - 1; j++) {
            state[j] = state[j + 1];
            action[j] = action[j + 1];
        }
        state[number_of_computers - 1] = state0;
        action[number_of_computers - 1] = action0;
    }

}

void fixed_constraints_local_search_scope_two(GRBModel& model, NumVar3D& alpha, NumArray3D& weight, int* state, int* action) {
    int current_state[number_of_computers], current_state_next[number_of_computers], current_action[number_of_computers];
    int state0, action0;
    for (int number = 0; number < number_of_computers; number++) {

        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = state[i];
            current_action[i] = action[i];
            current_state_next[i] = current_state[i];
        }

        for (int i = 0; i < number_of_computers; i++) {
            for (int j = 0; j < number_of_states; j++) {
                if (j != current_state[i]) {
                    current_state_next[i] = j;
                    if (compare_two_states_scope_two(weight, current_state, current_state_next, current_action) > 0) {
                        current_state[i] = j;
                    }
                }
            }
            current_state_next[i] = current_state[i];
        }
        add_constraint_to_master_problem_scope_two(model, alpha, current_state, current_action);

        state0 = state[0], action0 = action[0];
        for (int j = 0; j < number_of_computers - 1; j++) {
            state[j] = state[j + 1];
            action[j] = action[j + 1];
        }
        state[number_of_computers - 1] = state0;
        action[number_of_computers - 1] = action0;
    }

}

void fixed_constraints_local_search_scope_three(GRBModel& model, NumVar4D& alpha, NumArray4D& weight, int* state, int* action) {
    int current_state[number_of_computers], current_state_next[number_of_computers], current_action[number_of_computers];
    int state0, action0;
    for (int number = 0; number < number_of_computers; number++) {

        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = state[i];
            current_action[i] = action[i];
            current_state_next[i] = current_state[i];
        }

        for (int i = 0; i < number_of_computers; i++) {
            for (int j = 0; j < number_of_states; j++) {
                if (j != current_state[i]) {
                    current_state_next[i] = j;
                    if (compare_two_states_scope_three(weight, current_state, current_state_next, current_action) > 0) {
                        current_state[i] = j;
                    }
                }
            }
            current_state_next[i] = current_state[i];
        }
        add_constraint_to_master_problem_scope_three(model, alpha, current_state, current_action);

        state0 = state[0], action0 = action[0];
        for (int j = 0; j < number_of_computers - 1; j++) {
            state[j] = state[j + 1];
            action[j] = action[j + 1];
        }
        state[number_of_computers - 1] = state0;
        action[number_of_computers - 1] = action0;
    }

}



