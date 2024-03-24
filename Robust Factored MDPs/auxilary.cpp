#include "parameter.h"
#include "function.h"

extern int      neighbors[number_of_computers][number_of_neighbors];
extern double   prob_real[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern double   prob_robust[number_of_states][number_of_computers][number_of_constraints];

int decr(int i) {
    if (i > 0) return i - 1; else return number_of_computers - 1;
}

int incr(int i) {
    if (i < number_of_computers - 1) return i + 1; else return 0;
}

void generate_state_action(int* current_state, int* current_action) {
    double random_number;
    for (int j = 0; j < number_of_computers; ++j) {
        random_number = ((double)rand() / (RAND_MAX));
        if (random_number <= 1.0 / 3) current_state[j] = 0;
        else {
            if (random_number > 2.0 / 3) current_state[j] = 2;
            else current_state[j] = 1;
        }
        current_action[j] = 0;
    }
    for (int j = 0; j < number_of_actions; ++j) {
        random_number = ((double)rand() / (RAND_MAX));
        if (random_number == 1) random_number = 0.99;
        int action_i = (int)(number_of_computers * random_number);
        current_action[action_i] = 1;
    }
}

void generate_state_action_half(int* current_state, int* original_state, int* current_action, int& time) {
    double random_number;
    random_number = ((double)rand() / (RAND_MAX));
    if (random_number < 0.2) time = 0;
    else time = 1;
    for (int j = 0; j < number_of_computers; ++j) {
        random_number = ((double)rand() / (RAND_MAX));
        if (random_number <= 1.0 / 3) current_state[j] = 0;
        else {
            if (random_number > 2.0 / 3) current_state[j] = 2;
            else current_state[j] = 1;
        }
        current_action[j] = 0;
        original_state[j] = current_state[j];
    }

    for (int j = 0; j < number_of_actions; ++j) {
        random_number = ((double)rand() / (RAND_MAX));
        if (random_number == 1) random_number = 0.99;
        int action_i = (int)(number_of_computers * random_number);
        current_action[action_i] = 1;
        if ((action_i % 2 == 0) && (time == 1)) current_state[action_i] = 2;
    }
    if (time == 1) {
        for (int j = 0; j < number_of_computers; j += 2) {
            random_number = ((double)rand() / (RAND_MAX));
            if (random_number <= 1.0 / 3) original_state[j] = 0;
            else {
                if (random_number > 2.0 / 3) original_state[j] = 2;
                else original_state[j] = 1;
            }
        }
    }

}

double prob_out_robust(int* current_state, int i, int action_i, int state_i, int current_time, int constr) {
    double prob_i;
    if (i % 2 == current_time) {
        if (action_i == 1) {
            if (state_i == number_of_states - 1) prob_i = 1.0;
            else prob_i = 0.0;
        }
        else {
            prob_i = prob_robust[state_i][i][constr];
        }
    }
    else {
        if (state_i == current_state[i]) prob_i = 1.0;
        else prob_i = 0.0;
    }
    return prob_i;
}

double prob_out(int* current_state, int* original_state, int i, int action_i, int state_i, int current_time) {
    double prob_i;
    if (i % 2 == current_time) {
        if (action_i == 1) {
            if (state_i == number_of_states - 1) prob_i = 1.0;
            else prob_i = 0.0;
        }
        else {
            prob_i = prob_real[state_i][current_state[i]][original_state[neighbors[i][0]]][original_state[neighbors[i][1]]][i];
        }
    }
    else {
        if (state_i == current_state[i]) prob_i = 1.0;
        else prob_i = 0.0;
    }
    return prob_i;
}

