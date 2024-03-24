#include "parameter.h"
#include "function.h"

extern int      neighbors[number_of_computers][number_of_neighbors];
extern double   prob_real[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern double   prob_test[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern double   prob_hist[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern double   prob_robust[number_of_states][number_of_computers][number_of_constraints];
extern int      basis[number_of_computers][scope];
extern double   rewards[number_of_states][number_of_computers];


extern int      state_constr[number_of_computers][number_of_constraints];
extern int      original_constr[number_of_computers][number_of_constraints];
extern int      action_constr[number_of_computers][number_of_constraints];
extern int      time_constr[number_of_constraints];
extern double   epsilon;


double mastered_problem_weight(int constr_num, NumArray3D& weight, NumArray3D& weight_beta, NumArray3D& weight_gamma, NumArray2DA& weight_eta) {
    GRBEnv* env = new GRBEnv();
    GRBModel model = GRBModel(env);

    NumVar3D alpha, beta, gamma;
    NumVar2DA eta;
    GRBLinExpr obj_func;
    NumArray3D prob_sa;
    int current_state[number_of_computers], original_state[number_of_computers], current_action[number_of_computers];
    double prob_i, prob_j, sum, epsilon_h;
    int i, j;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_computers; k++) {
                alpha[i][j][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                beta[i][j][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                gamma[i][j][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                obj_func += alpha[i][j][k] / pow(number_of_states, scope);
            }
        }
    }

    for (int i = 0; i < number_of_computers; i++) {
        eta[0][i] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
        eta[1][i] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
    }


    model.setObjective(obj_func, GRB_MINIMIZE);

    std::cout << "master problem start ok" << endl;

    for (int constr = 0; constr < constr_num; constr++) {
        for (int k = 0; k < number_of_computers; k++) {
            current_state[k] = state_constr[k][constr];
            original_state[k] = original_constr[k][constr];
            current_action[k] = action_constr[k][constr];
        }
        if (time_constr[constr] == 0) {
            GRBLinExpr curr_LHS, curr_RHS;
            for (int k = 0; k < number_of_computers; k++)
                curr_LHS -= rewards[current_state[k]][k];
            for (int k = 0; k < number_of_computers; k++) {
                curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][k]
                    - gamma[original_state[basis[k][0]]][original_state[basis[k][1]]][k]
                    - eta[current_action[k]][k];
            }
            for (int k = 0; k < number_of_computers; k++) {
                sum = 0;
                for (int state_i = 0; state_i < number_of_states; ++state_i) {
                    for (int state_j = 0; state_j < number_of_states; state_j++) {
                        i = basis[k][0];
                        prob_i = prob_out_robust(current_state, i, current_action[i], state_i, 0, constr);
                        j = basis[k][1];
                        prob_j = prob_out_robust(current_state, j, current_action[j], state_j, 0, constr);
                        curr_RHS += prob_i * prob_j * beta[state_i][state_j][k];
                        sum += prob_i * prob_j;
                    }
                }
                //cout << sum << endl;
            }
            model.addConstr(curr_RHS - curr_LHS <= 0);
            curr_LHS.clear(); curr_RHS.clear();
        }
        else {
            GRBLinExpr curr_LHS, curr_RHS;
            for (int k = 0; k < number_of_computers; k++) {
                curr_LHS += beta[current_state[basis[k][0]]][current_state[basis[k][1]]][k]
                    + gamma[original_state[basis[k][0]]][original_state[basis[k][1]]][k]
                    + eta[current_action[k]][k];
            }
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                sum = 0;
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        i = basis[k][0];
                        prob_i = prob_out_robust(current_state, i, current_action[i], state_i, 1, constr);
                        j = basis[k][1];
                        prob_j = prob_out_robust(current_state, j, current_action[j], state_j, 1, constr);
                        curr_RHS += prob_i * prob_j * alpha[state_i][state_j][k];
                        sum += prob_i * prob_j;
                    }
                }
                //cout << sum << endl;
            }
            model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
            curr_LHS.clear(); curr_RHS.clear();
        }
    }

    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Threads, number_of_Threads);

    model.optimize();

    cout << "number of constraints:" << model.get(GRB_IntAttr_NumConstrs) << "  ";
    cout << "master problem tackled in : " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_computers; k++) {
                weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                weight_beta[i][j][k] = beta[i][j][k].get(GRB_DoubleAttr_X);
                weight_gamma[i][j][k] = gamma[i][j][k].get(GRB_DoubleAttr_X);
            }
        }
    }
    for (int i = 0; i < number_of_computers; i++) {
        weight_eta[0][i] = eta[0][i].get(GRB_DoubleAttr_X);
        weight_eta[1][i] = eta[1][i].get(GRB_DoubleAttr_X);
    }

    double obj_value = model.get(GRB_DoubleAttr_ObjVal);

    delete env;

    return obj_value;
}

double trans_prob_update(int constr, NumArray3D& weight, NumArray3D& weight_beta, int* current_state, int* current_action,
    int* original_state, int time) {

    int i, j;
    double prob_i;

    IloEnv env;
    IloModel model(env);

    typedef IloNumVar NumVar2DD[number_of_states][number_of_computers];
    NumVar2DD trans_prob;
    IloExpr obj_func(env);

    for (int k = 0; k < number_of_computers; k++) {
        for (int state_i = 0; state_i < number_of_states; ++state_i) {
            trans_prob[state_i][k] = IloNumVar(env, 0, 1.0);
        }
    }

    if (time == 0) {
        for (int k = 0; k < number_of_computers; k++) {
            i = basis[k][0];
            j = basis[k][1];
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                if (k % 2 == 0) obj_func += trans_prob[state_i][i] * weight_beta[state_i][current_state[j]][k];
                else obj_func += trans_prob[state_i][j] * weight_beta[current_state[i]][state_i][k];
            }
        }
    }
    else {
        for (int k = 0; k < number_of_computers; k++) {
            i = basis[k][0];
            j = basis[k][1];
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                if (k % 2 == 1) obj_func += trans_prob[state_i][i] * weight[state_i][current_state[j]][k];
                else obj_func += trans_prob[state_i][j] * weight[current_state[i]][state_i][k];
            }
        }
    }

    IloObjective objc = IloMinimize(env, obj_func);
    model.add(objc);
    obj_func.end();

    for (int i = 0; i < number_of_computers; i++) {
        IloExpr curr_LHS(env);
        for (int state_i = 0; state_i < number_of_states; ++state_i) {
            if (current_action[i] == 1) {
                if (state_i == number_of_states - 1) model.add(trans_prob[state_i][i] == 1);
                else model.add(trans_prob[state_i][i] == 0);
            }
            else {
                prob_i = prob_test[state_i][current_state[i]][original_state[neighbors[i][0]]][original_state[neighbors[i][1]]][i];
                model.add(trans_prob[state_i][i] <= prob_i + epsilon);
                model.add(trans_prob[state_i][i] >= prob_i - epsilon);
            }
            curr_LHS += trans_prob[state_i][i];
        }
        model.add(curr_LHS == 1.0);
        curr_LHS.end();
    }

    IloCplex cplex(model);
    cplex.setOut(env.getNullStream());
    cplex.solve();

    if (constr > -1) {
        for (int i = 0; i < number_of_computers; i++) {
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                prob_robust[state_i][i][constr] = cplex.getValue(trans_prob[state_i][i]);
            }
        }
    }


    double obj_value = cplex.getObjValue();

    env.end();

    return obj_value;
}


double mastered_problem_prob(int num_constr, NumArray3D& weight, NumArray3D& weight_beta) {

    int i, j;
    double prob_i, prob_j, curr_LHS, curr_RHS, obj_func;

    for (int constr = 0; constr < num_constr; constr++) {
        int current_state[number_of_computers], original_state[number_of_computers], current_action[number_of_computers];
        for (int k = 0; k < number_of_computers; k++) {
            current_state[k] = state_constr[k][constr];
            original_state[k] = original_constr[k][constr];
            current_action[k] = action_constr[k][constr];
        }
        curr_RHS = 0;
        if (time_constr[constr] == 0) {
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        i = basis[k][0];
                        prob_i = prob_out_robust(current_state, i, current_action[i], state_i, 0, constr);
                        j = basis[k][1];
                        prob_j = prob_out_robust(current_state, j, current_action[j], state_j, 0, constr);
                        curr_RHS += prob_i * prob_j * weight_beta[state_i][state_j][k];
                    }
                }
            }
        }
        else {
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        i = basis[k][0];
                        prob_i = prob_out_robust(current_state, i, current_action[i], state_i, 1, constr);
                        j = basis[k][1];
                        prob_j = prob_out_robust(current_state, j, current_action[j], state_j, 1, constr);
                        curr_RHS += prob_i * prob_j * weight[state_i][state_j][k];
                    }
                }
            }
        }
        curr_LHS = trans_prob_update(constr, weight, weight_beta, current_state, current_action, original_state, time_constr[constr]);

        obj_func = 0;
        if (time_constr[constr] == 0) {
            for (int k = 0; k < number_of_computers; k++) {
                i = basis[k][0];
                j = basis[k][1];
                for (int state_i = 0; state_i < number_of_states; ++state_i) {
                    if (k % 2 == 0) obj_func += prob_robust[state_i][i][constr] * weight_beta[state_i][current_state[j]][k];
                    else obj_func += prob_robust[state_i][j][constr] * weight_beta[current_state[i]][state_i][k];
                }
            }
        }
        else {
            for (int k = 0; k < number_of_computers; k++) {
                i = basis[k][0];
                j = basis[k][1];
                for (int state_i = 0; state_i < number_of_states; ++state_i) {
                    if (k % 2 == 1) obj_func += prob_robust[state_i][i][constr] * weight[state_i][current_state[j]][k];
                    else obj_func += prob_robust[state_i][j][constr] * weight[current_state[i]][state_i][k];
                }
            }
        }

        if (curr_LHS > curr_RHS + 1e-5) {
            cout << "error!!!!!!!" << endl;
            cout << "time: " << time_constr[constr] << endl;
            cout << "current states: ";
            for (i = 0; i < number_of_computers; i++) {
                cout << current_state[i];
            }
            cout << endl;
            cout << "current action: ";
            for (i = 0; i < number_of_computers; i++) {
                cout << current_action[i];
            }
            cout << endl;
            cout << "original state: ";
            for (i = 0; i < number_of_computers; i++) {
                cout << original_state[i];
            }
            cout << endl;

        }
    }
    return 0;
}


void creat_mastered_problem_NonRobust(GRBModel& model, NumVar3D& alpha) {
    GRBEnv env = model.getEnv();
    GRBLinExpr obj_func;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_computers; k++) {
                alpha[i][j][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                obj_func += alpha[i][j][k] / pow(number_of_states, scope);
            }
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Threads, number_of_Threads);
    model.update();
}

void add_constraint_to_master_problem(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action, int& constr_num) {
    GRBEnv env = model.getEnv();

    GRBLinExpr curr_LHS;
    for (int i = 0; i < number_of_computers; ++i)
        curr_LHS -= rewards[current_state[i]][i];
    for (int k = 0; k < number_of_computers; k++) {
        curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
    }

    GRBLinExpr curr_RHS;
    double prob_i, prob_j;
    int i, j;
    for (int state_i = 0; state_i < number_of_states; ++state_i) {
        for (int state_j = 0; state_j < number_of_states; state_j++) {
            for (int k = 0; k < number_of_computers; k++) {
                i = basis[k][0];
                if (current_action[i] == 1) {
                    if (state_i == number_of_states - 1) prob_i = 1.0;
                    else prob_i = 0.0;
                }
                else {
                    prob_i = prob_hist[state_i][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i];
                }
                j = basis[k][1];
                if (current_action[j] == 1) {
                    if (state_j == number_of_states - 1) prob_j = 1.0;
                    else prob_j = 0.0;
                }
                else {
                    prob_j = prob_hist[state_j][current_state[j]][current_state[neighbors[j][0]]][current_state[neighbors[j][1]]][j];
                }
                curr_RHS += prob_i * prob_j * alpha[state_i][state_j][k];
            }
        }
    }
    model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
    curr_LHS.clear(); curr_RHS.clear();

    //env.end();
}


void add_constraint_to_master_problem_true(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action) {
    GRBEnv env = model.getEnv();

    GRBLinExpr curr_LHS;
    for (int i = 0; i < number_of_computers; ++i)
        curr_LHS -= rewards[current_state[i]][i];
    for (int k = 0; k < number_of_computers; k++) {
        curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
    }

    GRBLinExpr curr_RHS;
    double prob_i, prob_j;
    int i, j;
    for (int state_i = 0; state_i < number_of_states; ++state_i) {
        for (int state_j = 0; state_j < number_of_states; state_j++) {
            for (int k = 0; k < number_of_computers; k++) {
                i = basis[k][0];
                if (current_action[i] == 1) {
                    if (state_i == number_of_states - 1) prob_i = 1.0;
                    else prob_i = 0.0;
                }
                else {
                    prob_i = prob_real[state_i][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i];
                }
                j = basis[k][1];
                if (current_action[j] == 1) {
                    if (state_j == number_of_states - 1) prob_j = 1.0;
                    else prob_j = 0.0;
                }
                else {
                    prob_j = prob_real[state_j][current_state[j]][current_state[neighbors[j][0]]][current_state[neighbors[j][1]]][j];
                }
                curr_RHS += prob_i * prob_j * alpha[state_i][state_j][k];
            }
        }
    }
    model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
    curr_LHS.clear(); curr_RHS.clear();

    //env.end();
}