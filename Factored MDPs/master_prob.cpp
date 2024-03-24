#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern int     basis[number_of_basis][scope];

void creat_mastered_problem_scope_one(GRBModel& model, NumVar2D& alpha) {
    GRBEnv env = model.getEnv();
    GRBLinExpr obj_func;

    for (int i = 0; i < number_of_states; i++) {
        for (int k = 0; k < number_of_basis; k++) {
            alpha[i][k] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
            obj_func += alpha[i][k] / pow(number_of_states, 1);
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.update();
}

void creat_mastered_problem_scope_two(GRBModel& model, NumVar3D& alpha) {
    GRBEnv env = model.getEnv();
    GRBLinExpr obj_func;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_basis; k++) {
                alpha[i][j][k] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                obj_func += alpha[i][j][k] / pow(number_of_states, 2);
            }
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.update();
}

void creat_mastered_problem_scope_three(GRBModel& model, NumVar4D& alpha) {
    GRBEnv env = model.getEnv();
    GRBLinExpr obj_func;

    for (int l = 0; l < number_of_states; l++) {
        for (int i = 0; i < number_of_states; i++) {
            for (int j = 0; j < number_of_states; j++) {
                for (int k = 0; k < number_of_basis; k++) {
                    alpha[l][i][j][k] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                    obj_func += alpha[l][i][j][k] / pow(number_of_states, 3);
                }
            }
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.update();
}

void add_constraint_to_master_problem_scope_one(GRBModel& model, NumVar2D& alpha,
    const int* current_state, const int* current_action) {
    GRBEnv env = model.getEnv();

    GRBLinExpr curr_LHS;
    for (int i = 0; i < number_of_computers; ++i)
        curr_LHS -= current_state[i];
    for (int k = 0; k < number_of_basis; k++) {
        curr_LHS += alpha[current_state[basis[k][0]]][k];
    }

    GRBLinExpr curr_RHS;
    double prob_i, prob_j;
    int i, j;
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
            curr_RHS += prob_i * alpha[state_i][k];
        }
    }
    model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
    curr_LHS.clear(); curr_RHS.clear();
    //env.end();
}

void add_constraint_to_master_problem_scope_two(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action) {
    GRBEnv env = model.getEnv();

    GRBLinExpr curr_LHS;
    for (int i = 0; i < number_of_computers; ++i)
        curr_LHS -= current_state[i];
    for (int k = 0; k < number_of_basis; k++) {
        curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
    }

    GRBLinExpr curr_RHS;
    double prob_i, prob_j;
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
                curr_RHS += prob_i * prob_j * alpha[state_i][state_j][k];
            }
        }
    }
    model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
    curr_LHS.clear(); curr_RHS.clear();
    //env.end();
}

void add_constraint_to_master_problem_scope_three(GRBModel& model, NumVar4D& alpha,
    const int* current_state, const int* current_action) {
    GRBEnv env = model.getEnv();

    GRBLinExpr curr_LHS;
    for (int i = 0; i < number_of_computers; ++i)
        curr_LHS -= current_state[i];
    for (int k = 0; k < number_of_basis; k++) {
        curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][current_state[basis[k][2]]][k];
    }

    GRBLinExpr curr_RHS;
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
                    curr_RHS += prob_i * prob_j * prob_l * alpha[state_i][state_j][state_l][k];
                }
            }
        }
    }
    model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
    curr_LHS.clear(); curr_RHS.clear();
    //env.end();
}

double solve_mastered_problem_scope_one(GRBModel& model, NumVar2D& alpha, NumArray2D& weight) {
    GRBEnv env = model.getEnv();

    model.getEnv().set(GRB_IntParam_OutputFlag, 0);
    model.getEnv().set(GRB_IntParam_Threads, number_of_Threads);
    model.getEnv().set(GRB_IntParam_Method, GRB_METHOD_DUAL);

    model.optimize();

    cout << "master problem tackled in : " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int k = 0; k < number_of_basis; k++) {
            weight[i][k] = alpha[i][k].get(GRB_DoubleAttr_X);
        }
    }

    return model.get(GRB_DoubleAttr_ObjVal);
    //env.end();
}

double solve_mastered_problem_scope_two(GRBModel& model, NumVar3D& alpha, NumArray3D& weight) {
    GRBEnv env = model.getEnv();

    model.getEnv().set(GRB_IntParam_OutputFlag, 0);
    model.getEnv().set(GRB_IntParam_Threads, number_of_Threads);
    model.getEnv().set(GRB_IntParam_Method, GRB_METHOD_DUAL);

    model.optimize();

    cout << "master problem tackled in : " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_basis; k++) {
                weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
            }
        }
    }

    return model.get(GRB_DoubleAttr_ObjVal);
    //env.end();
}

double solve_mastered_problem_scope_three(GRBModel& model, NumVar4D& alpha, NumArray4D& weight) {
    GRBEnv env = model.getEnv();

    model.getEnv().set(GRB_IntParam_OutputFlag, 0);
    model.getEnv().set(GRB_IntParam_Threads, number_of_Threads);
    model.getEnv().set(GRB_IntParam_Method, GRB_METHOD_DUAL);

    model.optimize();

    cout << "master problem tackled in : " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int l = 0; l < number_of_states; l++) {
                for (int k = 0; k < number_of_basis; k++) {
                    weight[i][j][l][k] = alpha[i][j][l][k].get(GRB_DoubleAttr_X);
                }
            }
        }
    }

    return model.get(GRB_DoubleAttr_ObjVal);
    //env.end();
}
