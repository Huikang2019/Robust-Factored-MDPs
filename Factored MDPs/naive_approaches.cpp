#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern int     basis[number_of_basis][scope];
extern int     type;
const int      number_of_repeat = 200, number_of_iter = 200;

void find_best_action_random(const int* current_state, int* best_action) {

    for (int j = 0; j < number_of_computers; ++j) {
        best_action[j] = 0;
    }
    vector<double> value_action;
    double prob_k, prob_z, value_k;
    for (int k = 0; k < number_of_computers; k++) {
        switch (current_state[k]) {
        case 0:
        case 1:
            value_action.push_back(2 + ((double)rand() / (RAND_MAX)));
            break;
        case 2:
            value_action.push_back(((double)rand() / (RAND_MAX)));
            break;
        }
    }
    vector<int> indices(value_action.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) { return value_action[i1] < value_action[i2]; });
    for (int j = 0; j < number_of_actions; j++) {
        best_action[indices[number_of_computers - 1 - j]] = 1;
    }
}

void find_best_action_priority(const int* current_state, int* best_action) {

    for (int j = 0; j < number_of_computers; ++j) {
        best_action[j] = 0;
    }
    vector<double> value_action;
    double prob_k, prob_z, value_k;
    for (int k = 0; k < number_of_computers; k++) {
        switch (current_state[k]) {
        case 0:
            value_action.push_back(2 + ((double)rand() / (RAND_MAX)));
            break;
        case 1:
            value_action.push_back(1 + ((double)rand() / (RAND_MAX)));
            break;
        case 2:
            value_action.push_back(((double)rand() / (RAND_MAX)));
            break;
        }
    }
    vector<int> indices(value_action.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) { return value_action[i1] < value_action[i2]; });
    for (int j = 0; j < number_of_actions; j++) {
        best_action[indices[number_of_computers - 1 - j]] = 1;
    }
}

void find_best_action_level(const int* current_state, int* best_action) {

    for (int j = 0; j < number_of_computers; ++j) {
        best_action[j] = 0;
    }
    vector<double> value_action;
    double prob_k, prob_z, value_k;
    if (type == 5) {
        for (int k = 0; k < number_of_computers; k++) {
            switch (current_state[k]) {
            case 0:
            case 1:
                if (k == 0) {
                    value_action.push_back(2 * 3 + ((double)rand() / (RAND_MAX)));
                    break;
                }
                if (2 * k < number_of_computers + 0.5) {
                    value_action.push_back(2 * 2 + ((double)rand() / (RAND_MAX)));
                    break;
                }
                if (2 * k > number_of_computers + 0.5) {
                    value_action.push_back(2 * 1 + ((double)rand() / (RAND_MAX)));
                    break;
                }
            case 2:
                value_action.push_back(((double)rand() / (RAND_MAX)));
                break;
            }
        }
    }
    if (type == 4) {
        for (int k = 0; k < number_of_computers; k++) {
            switch (current_state[k]) {
            case 0:
                value_action.push_back(4 * (3 - k % 3) + ((double)rand() / (RAND_MAX)));
                break;
            case 1:
                value_action.push_back((3 - k % 3) + ((double)rand() / (RAND_MAX)));
                break;
            case 2:
                value_action.push_back(((double)rand() / (RAND_MAX)));
                break;
            }
        }
    }
    if (type == 1) {
        for (int k = 0; k < number_of_computers; k++) {
            switch (current_state[k]) {
            case 0:
                value_action.push_back(2 + ((double)rand() / (RAND_MAX)));
                break;
            case 1:
                value_action.push_back(1 + ((double)rand() / (RAND_MAX)));
                break;
            case 2:
                value_action.push_back(((double)rand() / (RAND_MAX)));
                break;
            }
        }
    }
    vector<int> indices(value_action.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) { return value_action[i1] < value_action[i2]; });
    for (int j = 0; j < number_of_actions; j++) {
        best_action[indices[number_of_computers - 1 - j]] = 1;
    }
}

double monto_carlo_MIP_random(const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value, value_sum = 0;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i)
                value += current_state[i] * pow(discount_factor, iter);

            find_best_action_random(current_state, action_opt);

            for (int i = 0; i < number_of_computers; ++i) {
                if (action_opt[i] == 1) next_state[i] = 2;
                else {
                    random_number = ((double)rand() / (RAND_MAX));
                    if (random_number < prob[0][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]])
                        next_state[i] = 0;
                    else if (random_number > 1 - prob[2][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]])
                        next_state[i] = 2;
                    else next_state[i] = 1;
                }
            }

            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
    }

    return value_sum / (number_of_repeat * number_of_Threads);
}

double monto_carlo_MIP_priority(const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value, value_sum = 0;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i)
                value += current_state[i] * pow(discount_factor, iter);

            find_best_action_priority(current_state, action_opt);

            for (int i = 0; i < number_of_computers; ++i) {
                if (action_opt[i] == 1) next_state[i] = 2;
                else {
                    random_number = ((double)rand() / (RAND_MAX));
                    if (random_number < prob[0][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]])
                        next_state[i] = 0;
                    else if (random_number > 1 - prob[2][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]])
                        next_state[i] = 2;
                    else next_state[i] = 1;
                }
            }

            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
    }

    return value_sum / (number_of_repeat * number_of_Threads);
}

double monto_carlo_MIP_level(const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value, value_sum = 0;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i)
                value += current_state[i] * pow(discount_factor, iter);

            find_best_action_level(current_state, action_opt);

            for (int i = 0; i < number_of_computers; ++i) {
                if (action_opt[i] == 1) next_state[i] = 2;
                else {
                    random_number = ((double)rand() / (RAND_MAX));
                    if (random_number < prob[0][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]])
                        next_state[i] = 0;
                    else if (random_number > 1 - prob[2][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]])
                        next_state[i] = 2;
                    else next_state[i] = 1;
                }
            }
            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
    }

    return value_sum / (number_of_repeat * number_of_Threads);
}
