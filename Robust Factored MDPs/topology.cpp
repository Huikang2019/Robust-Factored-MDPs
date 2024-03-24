#include "parameter.h"
#include "function.h"

int     neighbors[number_of_computers][number_of_neighbors];
double  rewards[number_of_states][number_of_computers];
int     topology_basis[number_of_computers][2];
double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
double  prob_real[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
double  prob_robust[number_of_states][number_of_computers][number_of_constraints];
int     basis[number_of_computers][scope];
extern  double epsilon;

void create_tran_prob() {

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_states; k++) {
                prob[2][i][j][k] = 0.95 / ((double)(pow(1.5, 6 * (number_of_states - 1) - 4 * i - k - j)));
                if (prob[2][i][j][k] < 0.01) prob[2][i][j][k] = 0.0;
                prob[0][2 - i][2 - j][2 - k] = prob[2][i][j][k];
            }
        }
    }
    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_states; k++) {
                prob[1][i][j][k] = 1.0 - prob[2][i][j][k] - prob[0][i][j][k];
                if (prob[1][i][j][k] < 0) cout << "Transition probability: generation error!!!" << endl;
            }
        }
    }
    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_states; k++) {
                for (int h = 0; h < number_of_computers; h++) {                 
                    double random_number = ((double)rand() / (RAND_MAX));
                    random_number = 0.0;
                    prob_real[2][i][j][k][h] = prob[2][i][j][k] - epsilon * random_number;
                    prob_real[1][i][j][k][h] = prob[1][i][j][k];
                    if (prob_real[2][i][j][k][h] < 0) {
                        prob_real[2][i][j][k][h] = 0.0;
                        prob_real[1][i][j][k][h] = prob[1][i][j][k] - epsilon * random_number + prob[2][i][j][k];
                        if (prob_real[1][i][j][k][h] < 0) prob_real[1][i][j][k][h] = 0.0;
                    }
                    if (prob_real[2][i][j][k][h] > 1.0) {
                        prob_real[2][i][j][k][h] = 1.0;
                        prob_real[1][i][j][k][h] = 0.0;
                    }
                    if (prob_real[2][i][j][k][h] + prob_real[1][i][j][k][h] > 1.0) {
                        prob_real[1][i][j][k][h] = 1.0 - prob_real[2][i][j][k][h];
                    }
                    prob_real[0][i][j][k][h] = 1.0 - prob_real[2][i][j][k][h] - prob_real[1][i][j][k][h];
                    if (prob_real[0][i][j][k][h] < 0) cout << "Transition probability: generation error!!!" << endl;
                    //cout << "prob_2: " << prob_real[2][i][j][k][h] << " prob_1: " << prob_real[1][i][j][k][h] << " prob_0: " << prob_real[0][i][j][k][h] << endl;
                }

            }
        }
    }
}

// default: single-direct ring
// type 1:  bi-direct ring
// type 2:  star
// type 3:  ring and star
// type 4:  rings and rings
// type 5:  3 legs
void create_network_topology(int type) {
    double random_number[number_of_states], sum;
    for (int k = 0; k < number_of_computers; k++) {
        sum = 0;
        for (int i = 0; i < number_of_states; i++) {
            random_number[i] = ((double)rand() / (RAND_MAX));
            sum += random_number[i];
        }
        for (int i = 0; i < number_of_states; i++) {
            //rewards[i][k] = 3.0 * random_number[i] / sum;
            rewards[i][k] = pow(i, 2);
        }
    }

    switch (type) {
    default:
        for (int i = 0; i < number_of_computers; i++) {
            neighbors[i][0] = incr(i);
            neighbors[i][1] = i;
        }
        break;

    case 1:
        for (int i = 0; i < number_of_computers; i++) {
            neighbors[i][0] = incr(i);
            neighbors[i][1] = decr(i);
        }
        break;

    case 2:
        for (int i = 0; i < number_of_computers; i++) {
            neighbors[i][0] = 0;
            neighbors[i][1] = i;
        }
        break;

    case 3:
        for (int i = 0; i < number_of_computers; i++) {
            neighbors[i][0] = 0;
            if (i == 0) neighbors[i][1] = 0;
            else if (i == 1) neighbors[i][1] = number_of_computers - 1;
            else neighbors[i][1] = i - 1;
        }
        break;

    case 4:
        if (number_of_computers % 3 != 0) cout << "Can not generate a ring of rings !!!" << endl;

        for (int i = 0; i < number_of_computers; i++) {
            if (i % 3 == 0) {
                neighbors[i][0] = decr(decr(decr(i)));
                neighbors[i][1] = i;
            }
            else if (i % 3 == 1) {
                neighbors[i][0] = i - 1;
                neighbors[i][1] = i;
            }
            else {
                neighbors[i][0] = i - 2;
                neighbors[i][1] = i - 1;
            }
        }
        break;

    case 5:
        for (int i = 0; i < number_of_computers; i++) {
            neighbors[i][1] = i;
            if (i < 3) neighbors[i][0] = 0;
            else neighbors[i][0] = i - 3;
        }
        break;
    }

}

void create_basis_functions(int type) {
    switch (type) {
    default:
        for (int k = 0; k < number_of_computers; k++) {
            basis[k][0] = k;
            basis[k][1] = incr(k);
        }
        for (int k = number_of_computers; k < number_of_computers; k++) {
            basis[k][0] = k - number_of_computers;
            basis[k][1] = incr(incr(k - number_of_computers));
        }
        break;
    case 1:
        for (int k = 0; k < number_of_computers; k++) {
            basis[k][0] = k;
            basis[k][1] = incr(k);
        }
        for (int k = number_of_computers; k < number_of_computers; k++) {
            basis[k][0] = k - number_of_computers;
            basis[k][1] = incr(k - number_of_computers);
        }
        break;

    case 2:
    case 3:
        for (int k = 0; k < number_of_computers; k++) {
            basis[k][0] = k;
            basis[k][1] = incr(k);
        }
        for (int k = number_of_computers; k < number_of_computers; k++) {
            if (k == number_of_computers) {
                basis[k][0] = k - number_of_computers;
                basis[k][1] = 1;
            }
            else {
                basis[k][0] = k - number_of_computers;
                basis[k][1] = 0;
            }
        }
        break;
    case 4:
        for (int k = 0; k < number_of_computers; k++) {
            basis[k][0] = k;
            basis[k][1] = incr(k);
        }
        for (int k = number_of_computers; k < number_of_computers; k++) {
            if (k % 3 == 2) {
                basis[k][0] = k - number_of_computers;
                basis[k][1] = k - 2 - number_of_computers;
            }
            else {
                basis[k][0] = k - number_of_computers;
                basis[k][1] = incr(incr(incr(k - number_of_computers)));
            }
        }
        break;
    case 5:
        for (int k = 0; k < number_of_computers; k++) {
            if (k < 0) {
                basis[k][0] = 0;
                basis[k][1] = incr(k);
            }
            else {
                basis[k][0] = k;
                basis[k][1] = decr(decr(decr(k)));
                //basis[k][1] = incr(k);
            }

        }
        for (int k = number_of_computers; k < number_of_computers; k++) {
            basis[k][0] = k - number_of_computers;
            basis[k][1] = incr(incr(incr(k - number_of_computers)));
        }
        break;
    }
    int k, L;
    for (int i = 0; i < number_of_computers; i++) {
        k = basis[i][0];
        L = basis[i][1];
        int index = 0;
        if ((neighbors[k][0] != k) && (neighbors[k][0] != L)) {
            topology_basis[i][index] = neighbors[k][0];
            index++;
        }
        if ((neighbors[k][1] != k) && (neighbors[k][1] != L)
            && (neighbors[k][1] != neighbors[k][0])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis[i][index] = neighbors[k][1];
            index++;
        }
        if ((neighbors[L][0] != k) && (neighbors[L][0] != L)
            && (neighbors[L][0] != neighbors[k][0]) && (neighbors[L][0] != neighbors[k][1])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis[i][index] = neighbors[L][0];
            index++;
        }
        if ((neighbors[L][1] != k) && (neighbors[L][1] != L)
            && (neighbors[L][1] != neighbors[k][0]) && (neighbors[L][1] != neighbors[k][1])
            && (neighbors[L][1] != neighbors[L][0])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis[i][index] = neighbors[L][1];
            index++;
        }

    }
}