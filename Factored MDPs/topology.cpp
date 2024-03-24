#include <list>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "parameter.h"
#include "function.h"

using namespace std;

#define ILOUSESTL

int     neighbors[number_of_computers][number_of_neighbors];
int     topology_basis_scope_one[number_of_basis][2];
int     topology_basis_scope_two[number_of_basis][2];
int     topology_basis_scope_three[number_of_basis][2];
double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
int     basis[number_of_basis][scope];

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
                if (prob[1][i][j][k] <= 0) cout << "Transition probability: generation error!!!" << endl;
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

    for (int k = 0; k < number_of_computers; k++) {
        basis[k][0] = k;
        basis[k][1] = incr(k);
        basis[k][2] = decr(k);
    }

    int k;
    for (int i = 0; i < number_of_basis; i++) {
        k = basis[i][0];
        int index = 0;
        if ((neighbors[k][0] != k)) {
            topology_basis_scope_one[i][index] = neighbors[k][0];
            index++;
        }
        if ((neighbors[k][1] != k) && (neighbors[k][1] != neighbors[k][0])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_one[i][index] = neighbors[k][1];
            index++;
        }
    }

    int L;
    for (int i = 0; i < number_of_basis; i++) {
        k = basis[i][0];
        L = basis[i][1];
        int index = 0;
        if ((neighbors[k][0] != k) && (neighbors[k][0] != L)) {
            topology_basis_scope_two[i][index] = neighbors[k][0];
            index++;
        }
        if ((neighbors[k][1] != k) && (neighbors[k][1] != L)
            && (neighbors[k][1] != neighbors[k][0])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_two[i][index] = neighbors[k][1];
            index++;
        }
        if ((neighbors[L][0] != k) && (neighbors[L][0] != L) 
            && (neighbors[L][0] != neighbors[k][0]) && (neighbors[L][0] != neighbors[k][1])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_two[i][index] = neighbors[L][0];
            index++;
        }
        if ((neighbors[L][1] != k) && (neighbors[L][1] != L)
            && (neighbors[L][1] != neighbors[k][0]) && (neighbors[L][1] != neighbors[k][1])
            && (neighbors[L][1] != neighbors[L][0])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_two[i][index] = neighbors[L][1];
            index++;
        }
    }
    int h;
    for (int i = 0; i < number_of_basis; i++) {
        k = basis[i][0];
        L = basis[i][1];
        h = basis[i][2];
        int index = 0;
        if ((neighbors[k][0] != k) && (neighbors[k][0] != L) && (neighbors[k][0] != h)) {
            topology_basis_scope_three[i][index] = neighbors[k][0];
            index++;
        }
        if ((neighbors[k][1] != k) && (neighbors[k][1] != L) && (neighbors[k][1] != h)
            && (neighbors[k][1] != neighbors[k][0])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_three[i][index] = neighbors[k][1];
            index++;
        }
        if ((neighbors[L][0] != k) && (neighbors[L][0] != L) && (neighbors[L][0] != h)
            && (neighbors[L][0] != neighbors[k][0]) && (neighbors[L][0] != neighbors[k][1])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_three[i][index] = neighbors[L][0];
            index++;
        }
        if ((neighbors[L][1] != k) && (neighbors[L][1] != L) && (neighbors[L][1] != h)
            && (neighbors[L][1] != neighbors[k][0]) && (neighbors[L][1] != neighbors[k][1])
            && (neighbors[L][1] != neighbors[L][0])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_three[i][index] = neighbors[L][1];
            index++;
        }
        if ((neighbors[h][0] != k) && (neighbors[h][0] != L) && (neighbors[h][0] != h)
            && (neighbors[h][0] != neighbors[k][0]) && (neighbors[h][0] != neighbors[k][1])
            && (neighbors[h][0] != neighbors[L][0]) && (neighbors[h][0] != neighbors[L][1])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_three[i][index] = neighbors[h][0];
            index++;
        }
        if ((neighbors[h][1] != k) && (neighbors[h][1] != L) && (neighbors[h][1] != h)
            && (neighbors[h][1] != neighbors[k][0]) && (neighbors[h][1] != neighbors[k][1])
            && (neighbors[h][1] != neighbors[L][0]) && (neighbors[h][1] != neighbors[L][1])
            && (neighbors[h][1] != neighbors[h][0])) {
            if (index > 1) cout << "error!!" << endl;
            topology_basis_scope_three[i][index] = neighbors[h][1];
            index++;
        }

    }
}