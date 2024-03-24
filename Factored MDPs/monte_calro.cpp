#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern int     basis[number_of_basis][scope];
extern int     type;

const int      number_of_type = 3;
double         count_state[number_of_type][number_of_states];
double         count_action[number_of_type][number_of_states];
const int      number_of_repeat = 200, number_of_iter = 200;

double action_expected_value_scope_one(NumArray2D weight, const int* current_state, const int* current_action) {
    double curr_RHS = 0, prob_i;
    int i;
    for (int state_i = 0; state_i < number_of_states; ++state_i) {
        for (int k = 0; k < number_of_basis; k++) {
            i = basis[k][0];
            if (current_action[i] == 1) {
                if (state_i == number_of_states - 1) prob_i = 1.0;
                else prob_i = 0.0;
            }
            else {
                prob_i = prob[state_i][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]];
            }
            curr_RHS += prob_i * weight[state_i][k];
        }
    }
    return curr_RHS;
}

double action_expected_value_scope_two(NumArray3D weight, const int* current_state, const int* current_action) {
    double curr_RHS = 0, prob_i, prob_j;
    int i, j;
    for (int state_i = 0; state_i < number_of_states; ++state_i) {
        for (int state_j = 0; state_j < number_of_states; state_j++) {
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
                curr_RHS += prob_i * prob_j * weight[state_i][state_j][k];
            }
        }
    }
    return curr_RHS;
}

double action_expected_value_scope_three(NumArray4D weight, const int* current_state, const int* current_action) {
    double curr_RHS = 0, prob_i, prob_j, prob_l;
    int i, j, l;
    for (int state_i = 0; state_i < number_of_states; ++state_i) {
        for (int state_j = 0; state_j < number_of_states; state_j++) {
            for (int state_l = 0; state_l < number_of_states; state_l++) {
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
    return curr_RHS;
}

double find_best_action_milp_scope_one(NumArray2D weight, const int* current_state, int* best_action) {
    try {
        GRBEnv env = GRBEnv();
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model = GRBModel(env);

        // Create variables
        GRBVar* action = model.addVars(number_of_computers, GRB_BINARY);

        NumVarAction_one xi;
        int k;
        for (int ai = 0; ai < 2; ai++) {
            for (int index = 0; index < number_of_basis; index++) {
                k = basis[index][0];
                xi[ai][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                model.addConstr(xi[ai][index] <= (2 * ai - 1) * action[k] + 1 - ai);
                model.addConstr(xi[ai][index] >= 1 + (2 * ai - 1) * (action[k] - ai));
            }
        }

        GRBLinExpr sum_action = 0;
        for (int k = 0; k < number_of_computers; ++k)
            sum_action += action[k];
        model.addConstr(sum_action == number_of_actions);

        double prob_i, prob_j;
        int index_k1, index_k2, index_L1, index_L2;
        GRBLinExpr obj_func = 0;
        for (int i0 = 0; i0 < number_of_states; i0++) {
            for (int ai = 0; ai < 2; ai++) {
                for (int index = 0; index < number_of_basis; index++) {
                    k = basis[index][0];
                    if (ai == 1) {
                        if (i0 == number_of_states - 1) prob_i = 1.0;
                        else prob_i = 0.0;
                    }
                    else {
                        prob_i = prob[i0][current_state[k]][current_state[neighbors[k][0]]][current_state[neighbors[k][1]]];
                    }                  
                    obj_func += prob_i * weight[i0][index] * xi[ai][index];
                }
            }
        }


        model.setObjective(obj_func, GRB_MAXIMIZE);
        model.set(GRB_IntParam_Threads, number_of_Threads);
        model.set(GRB_IntParam_OutputFlag, 0);
        model.set(GRB_IntParam_Cuts, 1);
        model.optimize();

        // Get the optimal solution
        for (int i = 0; i < number_of_computers; i++) {
            if (action[i].get(GRB_DoubleAttr_X) > 0.5) best_action[i] = 1;
            else best_action[i] = 0;
        }

        double value = model.get(GRB_DoubleAttr_ObjVal);
        return value;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        return 0;
    }
}

double find_best_action_milp_scope_two(NumArray3D weight, const int* current_state, int* best_action) {
    try {
        GRBEnv env = GRBEnv();
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model = GRBModel(env);

        // Create variables
        GRBVar* action = model.addVars(number_of_computers, GRB_BINARY);

        NumVarAction_two xi;
        int k, L;
        for (int ai = 0; ai < 2; ai++) {
            for (int aj = 0; aj < 2; aj++) {
                for (int index = 0; index < number_of_basis; index++) {
                    k = basis[index][0];
                    L = basis[index][1];
                    xi[ai][aj][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                    model.addConstr(xi[ai][aj][index] <= (2 * ai - 1) * action[k] + 1 - ai);
                    model.addConstr(xi[ai][aj][index] <= (2 * aj - 1) * action[L] + 1 - aj);
                    model.addConstr(xi[ai][aj][index] >= 1 + (2 * ai - 1) * (action[k] - ai) + (2 * aj - 1) * (action[L] - aj));
                }
            }
        }

        GRBLinExpr sum_action = 0;
        for (int k = 0; k < number_of_computers; ++k)
            sum_action += action[k];
        model.addConstr(sum_action == number_of_actions);

        double prob_i, prob_j;
        int index_k1, index_k2, index_L1, index_L2;
        GRBLinExpr obj_func = 0;
        for (int i0 = 0; i0 < number_of_states; i0++) {
            for (int j0 = 0; j0 < number_of_states; j0++) {
                for (int ai = 0; ai < 2; ai++) {
                    for (int aj = 0; aj < 2; aj++) {
                        for (int index = 0; index < number_of_basis; index++) {
                            k = basis[index][0];
                            L = basis[index][1];
                            if (ai == 1) {
                                if (i0 == number_of_states - 1) prob_i = 1.0;
                                else prob_i = 0.0;
                            }
                            else {
                                prob_i = prob[i0][current_state[k]][current_state[neighbors[k][0]]][current_state[neighbors[k][1]]];
                            }
                            if (aj == 1) {
                                if (j0 == number_of_states - 1) prob_j = 1.0;
                                else prob_j = 0.0;
                            }
                            else {
                                prob_j = prob[j0][current_state[L]][current_state[neighbors[L][0]]][current_state[neighbors[L][1]]];
                            }
                            obj_func += prob_i * prob_j * weight[i0][j0][index] * xi[ai][aj][index];
                        }
                    }
                }
            }
        }


        model.setObjective(obj_func, GRB_MAXIMIZE);
        model.set(GRB_IntParam_Threads, number_of_Threads);
        model.set(GRB_IntParam_OutputFlag, 0);
        model.set(GRB_IntParam_Cuts, 1);
        model.optimize();

        // Get the optimal solution
        for (int i = 0; i < number_of_computers; i++) {
            if (action[i].get(GRB_DoubleAttr_X) > 0.5) best_action[i] = 1;
            else best_action[i] = 0;
        }

        double value = model.get(GRB_DoubleAttr_ObjVal);
        return value;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        return 0;
    }
}

double find_best_action_enumerate_scope_one(NumArray2D weight, const int* current_state, int* best_action) {

    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    switch (number_of_actions) {
    case 1:
        for (int i = 0; i < number_of_computers; i++) {
            current_action[i] = 1;
            current_value = action_expected_value_scope_one(weight, current_state, current_action);
            if (current_value > best_value) {
                for (int l = 0; l < number_of_computers; ++l) {
                    best_action[l] = current_action[l];
                }
                best_value = current_value;
            }
            current_action[i] = 0;
        }
        break;
    case 2:
        for (int i = 0; i < number_of_computers - 1; i++) {
            current_action[i] = 1;
            for (int j = i + 1; j < number_of_computers; j++) {
                current_action[j] = 1;
                current_value = action_expected_value_scope_one(weight, current_state, current_action);
                if (current_value > best_value) {
                    for (int l = 0; l < number_of_computers; ++l) {
                        best_action[l] = current_action[l];
                    }
                    best_value = current_value;
                }
                current_action[j] = 0;
            }
            current_action[i] = 0;
        }
        break;
    case 3:
        for (int i = 0; i < number_of_computers - 2; i++) {
            current_action[i] = 1;
            for (int j = i + 1; j < number_of_computers - 1; j++) {
                current_action[j] = 1;
                for (int k = j + 1; k < number_of_computers; k++) {
                    current_action[k] = 1;
                    current_value = action_expected_value_scope_one(weight, current_state, current_action);
                    if (current_value > best_value) {
                        for (int l = 0; l < number_of_computers; ++l) {
                            best_action[l] = current_action[l];
                        }
                        best_value = current_value;
                    }
                    current_action[k] = 0;
                }
                current_action[j] = 0;
            }
            current_action[i] = 0;
        }
        break;
    }
    
    return best_value;

}

double find_best_action_enumerate_scope_two(NumArray3D weight, const int* current_state, int* best_action) {

    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    switch (number_of_actions) {
    case 1:
        for (int i = 0; i < number_of_computers; i++) {
            current_action[i] = 1;
            current_value = action_expected_value_scope_two(weight, current_state, current_action);
            if (current_value > best_value) {
                for (int l = 0; l < number_of_computers; ++l) {
                    best_action[l] = current_action[l];
                }
                best_value = current_value;
            }
            current_action[i] = 0;
        }
        break;
    case 2:
        for (int i = 0; i < number_of_computers - 1; i++) {
            current_action[i] = 1;
            for (int j = i + 1; j < number_of_computers; j++) {
                current_action[j] = 1;
                current_value = action_expected_value_scope_two(weight, current_state, current_action);
                if (current_value > best_value) {
                    for (int l = 0; l < number_of_computers; ++l) {
                        best_action[l] = current_action[l];
                    }
                    best_value = current_value;
                }
                current_action[j] = 0;
            }
            current_action[i] = 0;
        }
        break;
    case 3:
        for (int i = 0; i < number_of_computers - 2; i++) {
            current_action[i] = 1;
            for (int j = i + 1; j < number_of_computers - 1; j++) {
                current_action[j] = 1;
                for (int k = j + 1; k < number_of_computers; k++) {
                    current_action[k] = 1;
                    current_value = action_expected_value_scope_two(weight, current_state, current_action);
                    if (current_value > best_value) {
                        for (int l = 0; l < number_of_computers; ++l) {
                            best_action[l] = current_action[l];
                        }
                        best_value = current_value;
                    }
                    current_action[k] = 0;
                }
                current_action[j] = 0;
            }
            current_action[i] = 0;
        }
        break;
    }

    return best_value;

}

double find_best_action_enumerate_scope_three(NumArray4D weight, const int* current_state, int* best_action) {

    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    switch (number_of_actions) {
    case 1:
        for (int i = 0; i < number_of_computers; i++) {
            current_action[i] = 1;
            current_value = action_expected_value_scope_three(weight, current_state, current_action);
            if (current_value > best_value) {
                for (int l = 0; l < number_of_computers; ++l) {
                    best_action[l] = current_action[l];
                }
                best_value = current_value;
            }
            current_action[i] = 0;
        }
        break;
    case 2:
        for (int i = 0; i < number_of_computers - 1; i++) {
            current_action[i] = 1;
            for (int j = i + 1; j < number_of_computers; j++) {
                current_action[j] = 1;
                current_value = action_expected_value_scope_three(weight, current_state, current_action);
                if (current_value > best_value) {
                    for (int l = 0; l < number_of_computers; ++l) {
                        best_action[l] = current_action[l];
                    }
                    best_value = current_value;
                }
                current_action[j] = 0;
            }
            current_action[i] = 0;
        }
        break;
    case 3:
        for (int i = 0; i < number_of_computers - 2; i++) {
            current_action[i] = 1;
            for (int j = i + 1; j < number_of_computers - 1; j++) {
                current_action[j] = 1;
                for (int k = j + 1; k < number_of_computers; k++) {
                    current_action[k] = 1;
                    current_value = action_expected_value_scope_three(weight, current_state, current_action);
                    if (current_value > best_value) {
                        for (int l = 0; l < number_of_computers; ++l) {
                            best_action[l] = current_action[l];
                        }
                        best_value = current_value;
                    }
                    current_action[k] = 0;
                }
                current_action[j] = 0;
            }
            current_action[i] = 0;
        }
        break;
    }

    return best_value;

}

double monto_carlo_MIP_scope_one(NumArray2D weight, const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    double value_sum = 0;
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i) {
                value += current_state[i] * pow(discount_factor, iter);
            }
            find_best_action_enumerate_scope_one(weight, current_state, action_opt);
            // find_best_action_milp_scope_one(weight, current_state, action_opt);


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
            switch (type) {
            case 1:
                for (int i = 0; i < number_of_computers; i++) {
                    count_state[0][current_state[i]] += 1.0 / number_of_computers;
                    if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0 / number_of_computers;
                }
                break;
            case 4:
                for (int i = 0; i < number_of_computers; i++) {
                    for (int j = 0; j < number_of_type; j++) {
                        if (i % 3 == j) {
                            count_state[j][current_state[i]] += 3.0 / number_of_computers;
                            if (action_opt[i] == 1) count_action[j][current_state[i]] += 3.0 / number_of_computers;
                        }
                    }
                }
                break;
            case 5:
                for (int i = 0; i < number_of_computers; i++) {
                    if (i == 0) {
                        count_state[0][current_state[i]] += 1.0;
                        if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0;
                    }
                    if (1 <= i && i <= number_of_computers / 2) {
                        count_state[1][current_state[i]] += 2.0 / number_of_computers;
                        if (action_opt[i] == 1) count_action[1][current_state[i]] += 2.0 / number_of_computers;
                    }
                    if (number_of_computers / 2 < i) {
                        count_state[2][current_state[i]] += 1.0 / (number_of_computers / 2.0 - 1);
                        if (action_opt[i] == 1) count_action[2][current_state[i]] += 1.0 / (number_of_computers / 2.0 - 1);
                    }
                }
                break;
            case 2:
            case 3:
                for (int i = 0; i < number_of_computers; i++) {
                    if (i == 0) {
                        count_state[0][current_state[i]] += 1.0;
                        if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0;
                    }
                    if (1 <= i) {
                        count_state[1][current_state[i]] += 1.0 / (number_of_computers - 1);
                        if (action_opt[i] == 1) count_action[1][current_state[i]] += 1.0 / (number_of_computers - 1);
                    }
                }
                break;
            }
            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
    }

    double value_mean = value_sum / (number_of_repeat * number_of_Threads);

    return value_mean;
}

double monto_carlo_MIP_scope_two(NumArray3D weight, const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    double value_sum = 0;
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i) {
                value += current_state[i] * pow(discount_factor, iter);
            }
            find_best_action_enumerate_scope_two(weight, current_state, action_opt);
            // find_best_action_milp_scope_two(weight, current_state, action_opt);


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
            switch (type) {
            case 1:
                for (int i = 0; i < number_of_computers; i++) {
                    count_state[0][current_state[i]] += 1.0 / number_of_computers;
                    if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0 / number_of_computers;
                }
                break;
            case 4:
                for (int i = 0; i < number_of_computers; i++) {
                    for (int j = 0; j < number_of_type; j++) {
                        if (i % 3 == j) {
                            count_state[j][current_state[i]] += 3.0 / number_of_computers;
                            if (action_opt[i] == 1) count_action[j][current_state[i]] += 3.0 / number_of_computers;
                        }
                    }
                }
                break;
            case 5:
                for (int i = 0; i < number_of_computers; i++) {
                    if (i == 0) {
                        count_state[0][current_state[i]] += 1.0;
                        if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0;
                    }
                    if (1 <= i && i <= number_of_computers / 2) {
                        count_state[1][current_state[i]] += 2.0 / number_of_computers;
                        if (action_opt[i] == 1) count_action[1][current_state[i]] += 2.0 / number_of_computers;
                    }
                    if (number_of_computers / 2 < i) {
                        count_state[2][current_state[i]] += 1.0 / (number_of_computers / 2.0 - 1);
                        if (action_opt[i] == 1) count_action[2][current_state[i]] += 1.0 / (number_of_computers / 2.0 - 1);
                    }
                }
                break;
            case 2:
            case 3:
                for (int i = 0; i < number_of_computers; i++) {
                    if (i == 0) {
                        count_state[0][current_state[i]] += 1.0;
                        if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0;
                    }
                    if (1 <= i) {
                        count_state[1][current_state[i]] += 1.0 / (number_of_computers - 1);
                        if (action_opt[i] == 1) count_action[1][current_state[i]] += 1.0 / (number_of_computers - 1);
                    }
                }
                break;
            }
            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
    }

    double value_mean = value_sum / (number_of_repeat * number_of_Threads);

    return value_mean;
}

double monto_carlo_MIP_scope_three(NumArray4D weight, const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    double value_sum = 0;
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i) {
                value += current_state[i] * pow(discount_factor, iter);
            }
            find_best_action_enumerate_scope_three(weight, current_state, action_opt);


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
            switch (type) {
            case 1:
                for (int i = 0; i < number_of_computers; i++) {
                    count_state[0][current_state[i]] += 1.0 / number_of_computers;
                    if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0 / number_of_computers;
                }
                break;
            case 4:
                for (int i = 0; i < number_of_computers; i++) {
                    for (int j = 0; j < number_of_type; j++) {
                        if (i % 3 == j) {
                            count_state[j][current_state[i]] += 3.0 / number_of_computers;
                            if (action_opt[i] == 1) count_action[j][current_state[i]] += 3.0 / number_of_computers;
                        }
                    }
                }
                break;
            case 5:
                for (int i = 0; i < number_of_computers; i++) {
                    if (i == 0) {
                        count_state[0][current_state[i]] += 1.0;
                        if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0;
                    }
                    if (1 <= i && i <= number_of_computers / 2) {
                        count_state[1][current_state[i]] += 2.0 / number_of_computers;
                        if (action_opt[i] == 1) count_action[1][current_state[i]] += 2.0 / number_of_computers;
                    }
                    if (number_of_computers / 2 < i) {
                        count_state[2][current_state[i]] += 1.0 / (number_of_computers / 2.0 - 1);
                        if (action_opt[i] == 1) count_action[2][current_state[i]] += 1.0 / (number_of_computers / 2.0 - 1);
                    }
                }
                break;
            case 2:
            case 3:
                for (int i = 0; i < number_of_computers; i++) {
                    if (i == 0) {
                        count_state[0][current_state[i]] += 1.0;
                        if (action_opt[i] == 1) count_action[0][current_state[i]] += 1.0;
                    }
                    if (1 <= i) {
                        count_state[1][current_state[i]] += 1.0 / (number_of_computers - 1);
                        if (action_opt[i] == 1) count_action[1][current_state[i]] += 1.0 / (number_of_computers - 1);
                    }
                }
                break;
            }
            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
    }

    double value_mean = value_sum / (number_of_repeat * number_of_Threads);

    return value_mean;
}




