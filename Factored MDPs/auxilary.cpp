#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];

int decr(int i) {
    if (i > 0) return i - 1; else return number_of_computers - 1;
}

int incr(int i) {
    if (i < number_of_computers - 1) return i + 1; else return 0;
}

void integer_to_binary(int inte, int* binary_i) {
    binary_i[0] = inte % 2;
    binary_i[1] = (inte - binary_i[0]) / 2;
}

int is_action_taken(int k, const int* current_action) {
    int yes = 0;
    for (int i = 0; i < number_of_actions; i++) {
        if (k == current_action[i]) {
            yes = 1;
            break;
        }
    }
    return yes;
}

int state_to_index(int* current_state) {
    int state_index = 0;
    for (int k = 0; k < number_of_computers; ++k) {
        state_index += current_state[k] * pow(number_of_states, k);
    }
    return state_index;
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