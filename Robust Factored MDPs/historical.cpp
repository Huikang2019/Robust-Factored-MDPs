#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern int     topology_basis[number_of_computers][2];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern double  prob_real[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern int     basis[number_of_computers][scope];
extern int     type;
extern double  rewards[number_of_states][number_of_computers];
extern double  prob_robust[number_of_states][number_of_computers][number_of_constraints];
extern double  test_ratio;
extern int     number_of_history_data;

double  transition_count_hist[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
double  transition_count_test[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
double  transition_count_validation[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
double  prob_test[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
double  prob_validation[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
double  prob_hist[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];

double random_action(const int* current_state, int* best_action) {

    double random_number[number_of_computers], random_number_old[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        random_number_old[i] = ((double)rand() / (RAND_MAX)) + 1.0 * current_state[i];
        random_number[i] = random_number_old[i];
        best_action[i] = 0;
    }
    sort(random_number, random_number + number_of_computers);
    for (int k = 0; k < number_of_actions; k++) {
        for (int i = 0; i < number_of_computers; ++i) {
            if (random_number_old[i] == random_number[k]) best_action[i] = 1;
        }
    }
    return 0;
}


void monto_carlo_history() {

    double random_number, value_opt = 0, value_k = 0, value, random_number_gen, random_number_cor;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    generate_state_action(current_state, action_opt);
    value = 0;
    for (int iter = 0; iter < number_of_history_data; ++iter) {
        for (int i = 0; i < number_of_computers; ++i)
            value += rewards[current_state[i]][i] * pow(discount_factor, iter);

        random_action(current_state, action_opt);

        random_number_gen = ((double)rand() / (RAND_MAX));
        random_number_cor = ((double)rand() / (RAND_MAX));
        for (int i = 0; i < number_of_computers; ++i) {
            if (random_number_cor < correlation) random_number = random_number_gen;
            else random_number = ((double)rand() / (RAND_MAX));
            if (action_opt[i] == 1) next_state[i] = 2;
            else {
                if (random_number < prob_real[0][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i])
                    next_state[i] = 0;
                else if (random_number > 1 - prob_real[2][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i])
                    next_state[i] = 2;
                else next_state[i] = 1;
                transition_count_hist[next_state[i]][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i]++;
                if (((double)rand() / (RAND_MAX)) < test_ratio) transition_count_test[next_state[i]][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i]++;
                else transition_count_validation[next_state[i]][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i]++;
            }
        }
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = next_state[i];
        }
    }
    double sum_tran_hist, sum_tran_test, sum_tran_validation;
    for (int g = 0; g < number_of_computers; g++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_states; k++) {
                for (int h = 0; h < number_of_states; h++) {
                    sum_tran_hist = 0;
                    sum_tran_test = 0;
                    sum_tran_validation = 0;
                    for (int i = 0; i < number_of_states; i++) {
                        sum_tran_hist += transition_count_hist[i][j][k][h][g];
                        sum_tran_test += transition_count_test[i][j][k][h][g];
                        sum_tran_validation += transition_count_validation[i][j][k][h][g];
                    }
                    for (int i = 0; i < number_of_states; i++) {
                        if (sum_tran_hist > 0) prob_hist[i][j][k][h][g] = transition_count_hist[i][j][k][h][g] / sum_tran_hist;
                        else prob_hist[i][j][k][h][g] = 1.0 / number_of_states;
                        if (sum_tran_test > 0) prob_test[i][j][k][h][g] = transition_count_test[i][j][k][h][g] / sum_tran_test;
                        else prob_test[i][j][k][h][g] = 1.0 / number_of_states;
                        if (sum_tran_validation > 0) prob_validation[i][j][k][h][g] = transition_count_validation[i][j][k][h][g] / sum_tran_validation;
                        else prob_validation[i][j][k][h][g] = 1.0 / number_of_states;
                        //if (sum_tran_hist > 0) cout << prob_test[i][j][k][h][g] << " vs " << prob_hist[i][j][k][h][g] << " vs " << prob_real[i][j][k][h][g] << " vs " << prob_validation[i][j][k][h][g] << endl;
                    }
                }
            }
        }
    }
}

