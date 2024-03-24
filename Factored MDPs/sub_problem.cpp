#include "parameter.h"
#include "function.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <exception>

extern int     neighbors[number_of_computers][number_of_neighbors];
extern int     topology_basis_scope_one[number_of_basis][2];
extern int     topology_basis_scope_two[number_of_basis][2];
extern int     topology_basis_scope_three[number_of_basis][2];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern int     basis[number_of_basis][scope];

class MyCallback : public GRBCallback {
private:
    double stopThreshold;
    double minRuntim;

public:
    MyCallback(double threshold, double runtim) : stopThreshold(threshold), minRuntim(runtim) {}

    void callback() {
        if (where == GRB_CB_MIPSOL) {
            if ((getDoubleInfo(GRB_CB_RUNTIME) > minRuntim) &&
                (getDoubleInfo(GRB_CB_MIPSOL_OBJ) < stopThreshold)) {
                abort();
            }
        }
    }
};

double solve_subed_problem_GRB_scope_one(NumArray2D weight, double eps, double time_limit_least, 
    double time_limit_most, int* worst_state, int* worst_action) {
    GRBEnv* env = new GRBEnv();

    GRBModel model = GRBModel(env);

    // decision variables: alpha_1k, alpha_2k
    GRBVar* state = model.addVars(2 * number_of_computers, GRB_BINARY);

    GRBVar* action = model.addVars(number_of_computers, GRB_BINARY);

    GRBLinExpr obj_func_1, obj_func_2, curr_RHS;
    for (int k = 0; k < number_of_computers; ++k)
        obj_func_1 -= state[2 * k] + 2 * state[2 * k + 1];

    typedef GRBVar NumVar2D_GRB[number_of_states][number_of_basis];
    NumVar2D_GRB beta; // auxiliary variables for
    int binary_i[2];
    int k;
    for (int i = 0; i < number_of_states; i++) {
        integer_to_binary(i, binary_i);
        for (int index = 0; index < number_of_basis; index++) {
            k = basis[index][0];
            beta[i][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
            obj_func_1 += beta[i][index] * weight[i][index];
            curr_RHS = (state[2 * k] - binary_i[0]) * (2 * binary_i[0] - 1);
            curr_RHS += (state[2 * k + 1] - binary_i[1]) * (2 * binary_i[1] - 1);
            model.addConstr(beta[i][index] >= 1 + curr_RHS);
            model.addConstr(beta[i][index] <= (2 * binary_i[0] - 1) * state[2 * k] + 1 - binary_i[0]);
            model.addConstr(beta[i][index] <= (2 * binary_i[1] - 1) * state[2 * k + 1] + 1 - binary_i[1]);
        }
    }

    typedef GRBVar NumVar5D_GRB[number_of_states][number_of_states][number_of_states][2][number_of_basis];
    NumVar5D_GRB xi;     // auxiliary variables for g
    int binary_i1[2], binary_j1[2], binary_i2[2], binary_j2[2], binary_i3[2], binary_j3[2];
    for (int i1 = 0; i1 < number_of_states; i1++) {
        integer_to_binary(i1, binary_i1);
        for (int i2 = 0; i2 < number_of_states; i2++) {
            integer_to_binary(i2, binary_i2);
            for (int j2 = 0; j2 < number_of_states; j2++) {
                integer_to_binary(j2, binary_j2);
                for (int ai = 0; ai < 2; ai++) {
                    for (int index = 0; index < number_of_basis; index++) {
                        k = basis[index][0];
                        xi[i1][i2][j2][ai][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                        model.addConstr(xi[i1][i2][j2][ai][index] <= (2 * binary_i1[0] - 1) * state[2 * k] + 1 - binary_i1[0]);
                        model.addConstr(xi[i1][i2][j2][ai][index] <= (2 * binary_i1[1] - 1) * state[2 * k + 1] + 1 - binary_i1[1]);
                        model.addConstr(xi[i1][i2][j2][ai][index] <= (2 * binary_i2[0] - 1) * state[2 * topology_basis_scope_one[index][0]] + 1 - binary_i2[0]);
                        model.addConstr(xi[i1][i2][j2][ai][index] <= (2 * binary_i2[1] - 1) * state[2 * topology_basis_scope_one[index][0] + 1] + 1 - binary_i2[1]);
                        model.addConstr(xi[i1][i2][j2][ai][index] <= (2 * binary_j2[0] - 1) * state[2 * topology_basis_scope_one[index][1]] + 1 - binary_j2[0]);
                        model.addConstr(xi[i1][i2][j2][ai][index] <= (2 * binary_j2[1] - 1) * state[2 * topology_basis_scope_one[index][1] + 1] + 1 - binary_j2[1]);
                        model.addConstr(xi[i1][i2][j2][ai][index] <= (2 * ai - 1) * action[k] + 1 - ai);

                        curr_RHS = (2 * binary_i1[0] - 1) * (state[2 * k] - binary_i1[0]);
                        curr_RHS += (2 * binary_i1[1] - 1) * (state[2 * k + 1] - binary_i1[1]);
                        curr_RHS += (2 * binary_i2[0] - 1) * (state[2 * topology_basis_scope_one[index][0]] - binary_i2[0]);
                        curr_RHS += (2 * binary_i2[1] - 1) * (state[2 * topology_basis_scope_one[index][0] + 1] - binary_i2[1]);
                        curr_RHS += (2 * binary_j2[0] - 1) * (state[2 * topology_basis_scope_one[index][1]] - binary_j2[0]);
                        curr_RHS += (2 * binary_j2[1] - 1) * (state[2 * topology_basis_scope_one[index][1] + 1] - binary_j2[1]);
                        curr_RHS += (2 * ai - 1) * (action[k] - ai);
                        model.addConstr(xi[i1][i2][j2][ai][index] >= 1 + curr_RHS);
                    }
                }
            }
        }
    }


    double prob_i;
    int index_k1, index_k2;
    for (int i0 = 0; i0 < number_of_states; i0++) {
        for (int i1 = 0; i1 < number_of_states; i1++) {
            for (int i2 = 0; i2 < number_of_states; i2++) {
                for (int j2 = 0; j2 < number_of_states; j2++) {
                    for (int ai = 0; ai < 2; ai++) {
                        for (int index = 0; index < number_of_basis; index++) {
                            k = basis[index][0];
                            if (ai == 1) {
                                if (i0 == number_of_states - 1) prob_i = 1.0;
                                else prob_i = 0.0;
                            }
                            else {
                                if (neighbors[k][0] == k) index_k1 = i1;
                                if (neighbors[k][0] == topology_basis_scope_one[index][0]) index_k1 = i2;
                                if (neighbors[k][0] == topology_basis_scope_one[index][1]) index_k1 = j2;
                                if (neighbors[k][1] == k) index_k2 = i1;
                                if (neighbors[k][1] == topology_basis_scope_one[index][0]) index_k2 = i2;
                                if (neighbors[k][1] == topology_basis_scope_one[index][1]) index_k2 = j2;
                                prob_i = prob[i0][i1][index_k1][index_k2];
                            }
                            obj_func_2 += prob_i * weight[i0][index] * xi[i1][i2][j2][ai][index];
                        }
                    }
                }
            }
        }
    }

    GRBLinExpr obj = obj_func_1 - discount_factor * obj_func_2;
    model.setObjective(obj, GRB_MINIMIZE);

    // generate the constraints: x0[k] + x1[k] + x2[k] = 1 for all k
    GRBLinExpr sum_action;
    for (int k = 0; k < number_of_computers; ++k) sum_action += action[k];
    model.addConstr(sum_action == number_of_actions);

    for (int k = 0; k < number_of_computers; ++k) model.addConstr(state[2 * k] + 2 * state[2 * k + 1] <= number_of_states - 1);

    // solve the problem
    model.set(GRB_IntParam_Threads, number_of_Threads);
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Cuts, 1);
    env->set(GRB_DoubleParam_Cutoff, eps);

    // 进行热启动
    model.getEnv().set(GRB_IntParam_Presolve, 0);
    MyCallback cb(eps, time_limit_least);
    model.setCallback(&cb);

    model.getEnv().set(GRB_DoubleParam_TimeLimit, time_limit_most);

    model.optimize();

    double objValue = model.get(GRB_DoubleAttr_ObjVal);
    double bestBound = model.get(GRB_DoubleAttr_ObjBound);
    double gap = model.get(GRB_DoubleAttr_MIPGap);
    // std::cout << "Objective function value: " << objValue << std::endl;
    // std::cout << "Best bound: " << bestBound << std::endl;
    // std::cout << "Gap: " << gap << std::endl;

    // cout << "*** SLAVE OBJECTIVE VALUE: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    // cout << "sub problem tackled in : " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    // add constraint to master problem if necessary
    for (int i = 0; i < number_of_computers; i++) {
        worst_state[i] = state[2 * i].get(GRB_DoubleAttr_X) + 2 * state[2 * i + 1].get(GRB_DoubleAttr_X);
        if (action[i].get(GRB_DoubleAttr_X) > 0.5) worst_action[i] = 1;
        else worst_action[i] = 0;
    }

    // that's it!
    double value = model.get(GRB_DoubleAttr_ObjVal);

    delete env;

    return value;
}

double solve_subed_problem_GRB_scope_two(NumArray3D weight, double eps, double time_limit_least, double time_limit_most, int* worst_state, int* worst_action) {
    GRBEnv* env = new GRBEnv();

    GRBModel model = GRBModel(env);

    // decision variables: alpha_1k, alpha_2k
    GRBVar* state = model.addVars(2 * number_of_computers, GRB_BINARY);

    GRBVar* action = model.addVars(number_of_computers, GRB_BINARY);

    GRBLinExpr obj_func_1, obj_func_2, curr_RHS;
    for (int k = 0; k < number_of_computers; ++k)
        obj_func_1 -= state[2 * k] + 2 * state[2 * k + 1];

    typedef GRBVar NumVar3D_GRB[number_of_states][number_of_states][number_of_basis];
    NumVar3D_GRB beta; // auxiliary variables for
    int binary_i[2], binary_j[2];
    int k, L;
    for (int i = 0; i < number_of_states; i++) {
        integer_to_binary(i, binary_i);
        for (int j = 0; j < number_of_states; j++) {
            integer_to_binary(j, binary_j);
            for (int index = 0; index < number_of_basis; index++) {
                k = basis[index][0];
                L = basis[index][1];
                beta[i][j][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                obj_func_1 += beta[i][j][index] * weight[i][j][index];
                curr_RHS = (state[2 * k] - binary_i[0]) * (2 * binary_i[0] - 1);
                curr_RHS += (state[2 * k + 1] - binary_i[1]) * (2 * binary_i[1] - 1);
                curr_RHS += (state[2 * L] - binary_j[0]) * (2 * binary_j[0] - 1);
                curr_RHS += (state[2 * L + 1] - binary_j[1]) * (2 * binary_j[1] - 1);
                model.addConstr(beta[i][j][index] >= 1 + curr_RHS);
                model.addConstr(beta[i][j][index] <= (2 * binary_i[0] - 1) * state[2 * k] + 1 - binary_i[0]);
                model.addConstr(beta[i][j][index] <= (2 * binary_i[1] - 1) * state[2 * k + 1] + 1 - binary_i[1]);
                model.addConstr(beta[i][j][index] <= (2 * binary_j[0] - 1) * state[2 * L] + 1 - binary_j[0]);
                model.addConstr(beta[i][j][index] <= (2 * binary_j[1] - 1) * state[2 * L + 1] + 1 - binary_j[1]);
            }
        }
    }

    typedef GRBVar NumVar8D_GRB[number_of_states][number_of_states][number_of_states][number_of_states][2][2][number_of_basis];
    NumVar8D_GRB xi;     // auxiliary variables for g
    int binary_i1[2], binary_j1[2], binary_i2[2], binary_j2[2], binary_i3[2], binary_j3[2];
    for (int i1 = 0; i1 < number_of_states; i1++) {
        integer_to_binary(i1, binary_i1);
        for (int j1 = 0; j1 < number_of_states; j1++) {
            integer_to_binary(j1, binary_j1);
            for (int i2 = 0; i2 < number_of_states; i2++) {
                integer_to_binary(i2, binary_i2);
                for (int j2 = 0; j2 < number_of_states; j2++) {
                    integer_to_binary(j2, binary_j2);
                    for (int ai = 0; ai < 2; ai++) {
                        for (int aj = 0; aj < 2; aj++) {
                            for (int index = 0; index < number_of_basis; index++) {
                                k = basis[index][0];
                                L = basis[index][1];
                                xi[i1][j1][i2][j2][ai][aj][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * binary_i1[0] - 1) * state[2 * k] + 1 - binary_i1[0]);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * binary_i1[1] - 1) * state[2 * k + 1] + 1 - binary_i1[1]);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * binary_j1[0] - 1) * state[2 * L] + 1 - binary_j1[0]);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * binary_j1[1] - 1) * state[2 * L + 1] + 1 - binary_j1[1]);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * binary_i2[0] - 1) * state[2 * topology_basis_scope_two[index][0]] + 1 - binary_i2[0]);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * binary_i2[1] - 1) * state[2 * topology_basis_scope_two[index][0] + 1] + 1 - binary_i2[1]);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * binary_j2[0] - 1) * state[2 * topology_basis_scope_two[index][1]] + 1 - binary_j2[0]);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * binary_j2[1] - 1) * state[2 * topology_basis_scope_two[index][1] + 1] + 1 - binary_j2[1]);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * ai - 1) * action[k] + 1 - ai);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] <= (2 * aj - 1) * action[L] + 1 - aj);

                                curr_RHS = (2 * binary_i1[0] - 1) * (state[2 * k] - binary_i1[0]);
                                curr_RHS += (2 * binary_i1[1] - 1) * (state[2 * k + 1] - binary_i1[1]);
                                curr_RHS += (2 * binary_j1[0] - 1) * (state[2 * L] - binary_j1[0]);
                                curr_RHS += (2 * binary_j1[1] - 1) * (state[2 * L + 1] - binary_j1[1]);
                                curr_RHS += (2 * binary_i2[0] - 1) * (state[2 * topology_basis_scope_two[index][0]] - binary_i2[0]);
                                curr_RHS += (2 * binary_i2[1] - 1) * (state[2 * topology_basis_scope_two[index][0] + 1] - binary_i2[1]);
                                curr_RHS += (2 * binary_j2[0] - 1) * (state[2 * topology_basis_scope_two[index][1]] - binary_j2[0]);
                                curr_RHS += (2 * binary_j2[1] - 1) * (state[2 * topology_basis_scope_two[index][1] + 1] - binary_j2[1]);
                                curr_RHS += (2 * ai - 1) * (action[k] - ai);
                                curr_RHS += (2 * aj - 1) * (action[L] - aj);
                                model.addConstr(xi[i1][j1][i2][j2][ai][aj][index] >= 1 + curr_RHS);
                            }
                        }
                    }
                }
            }
        }
    }

    double prob_i, prob_j;
    int index_k1, index_k2, index_L1, index_L2;
    for (int i0 = 0; i0 < number_of_states; i0++) {
        for (int j0 = 0; j0 < number_of_states; j0++) {
            for (int i1 = 0; i1 < number_of_states; i1++) {
                for (int j1 = 0; j1 < number_of_states; j1++) {
                    for (int i2 = 0; i2 < number_of_states; i2++) {
                        for (int j2 = 0; j2 < number_of_states; j2++) {
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
                                            if (neighbors[k][0] == k) index_k1 = i1;
                                            if (neighbors[k][0] == L) index_k1 = j1;
                                            if (neighbors[k][0] == topology_basis_scope_two[index][0]) index_k1 = i2;
                                            if (neighbors[k][0] == topology_basis_scope_two[index][1]) index_k1 = j2;
                                            if (neighbors[k][1] == k) index_k2 = i1;
                                            if (neighbors[k][1] == L) index_k2 = j1;
                                            if (neighbors[k][1] == topology_basis_scope_two[index][0]) index_k2 = i2;
                                            if (neighbors[k][1] == topology_basis_scope_two[index][1]) index_k2 = j2;
                                            prob_i = prob[i0][i1][index_k1][index_k2];
                                        }
                                        if (aj == 1) {
                                            if (j0 == number_of_states - 1) prob_j = 1.0;
                                            else prob_j = 0.0;
                                        }
                                        else {
                                            if (neighbors[L][0] == k) index_k1 = i1;
                                            if (neighbors[L][0] == L) index_k1 = j1;
                                            if (neighbors[L][0] == topology_basis_scope_two[index][0]) index_k1 = i2;
                                            if (neighbors[L][0] == topology_basis_scope_two[index][1]) index_k1 = j2;
                                            if (neighbors[L][1] == k) index_k2 = i1;
                                            if (neighbors[L][1] == L) index_k2 = j1;
                                            if (neighbors[L][1] == topology_basis_scope_two[index][0]) index_k2 = i2;
                                            if (neighbors[L][1] == topology_basis_scope_two[index][1]) index_k2 = j2;
                                            prob_j = prob[j0][j1][index_k1][index_k2];
                                        }
                                        obj_func_2 += prob_i * prob_j * weight[i0][j0][index] * xi[i1][j1][i2][j2][ai][aj][index];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    GRBLinExpr obj = obj_func_1 - discount_factor * obj_func_2;
    model.setObjective(obj, GRB_MINIMIZE);

    // generate the constraints: x0[k] + x1[k] + x2[k] = 1 for all k
    GRBLinExpr sum_action;
    for (int k = 0; k < number_of_computers; ++k) sum_action += action[k];
    model.addConstr(sum_action == number_of_actions);

    for (int k = 0; k < number_of_computers; ++k) model.addConstr(state[2 * k] + 2 * state[2 * k + 1] <= number_of_states - 1);

    // solve the problem
    model.set(GRB_IntParam_Threads, number_of_Threads);
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Cuts, 1);
    env->set(GRB_DoubleParam_Cutoff, eps);

    // 进行热启动
    model.getEnv().set(GRB_IntParam_Presolve, 0);
    MyCallback cb(eps, time_limit_least);
    model.setCallback(&cb);

    model.getEnv().set(GRB_DoubleParam_TimeLimit, time_limit_most);

    model.optimize();

    double objValue = model.get(GRB_DoubleAttr_ObjVal);
    double bestBound = model.get(GRB_DoubleAttr_ObjBound);
    double gap = model.get(GRB_DoubleAttr_MIPGap);
    // std::cout << "Objective function value: " << objValue << std::endl;
    // std::cout << "Best bound: " << bestBound << std::endl;
    // std::cout << "Gap: " << gap << std::endl;

    // cout << "*** SLAVE OBJECTIVE VALUE: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    // cout << "sub problem tackled in : " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    // add constraint to master problem if necessary
    for (int i = 0; i < number_of_computers; i++) {
        worst_state[i] = state[2 * i].get(GRB_DoubleAttr_X) + 2 * state[2 * i + 1].get(GRB_DoubleAttr_X);
        if (action[i].get(GRB_DoubleAttr_X) > 0.5) worst_action[i] = 1;
        else worst_action[i] = 0;
    }

    // that's it!
    double value = model.get(GRB_DoubleAttr_ObjVal);

    delete env;

    return value;
}

double solve_subed_problem_GRB_scope_three(NumArray4D weight, double eps, double time_limit_least, double time_limit_most, int* worst_state, int* worst_action) {
    GRBEnv* env = new GRBEnv();
    GRBModel model = GRBModel(env);

    // decision variables: alpha_1k, alpha_2k
    GRBVar* state = model.addVars(2 * number_of_computers, GRB_BINARY);

    GRBVar* action = model.addVars(number_of_computers, GRB_BINARY);

    GRBLinExpr obj_func_1, obj_func_2, curr_RHS;
    for (int k = 0; k < number_of_computers; ++k)
        obj_func_1 -= state[2 * k] + 2 * state[2 * k + 1];

    typedef GRBVar NumVar4D_GRB[number_of_states][number_of_states][number_of_states][number_of_basis];
    NumVar4D_GRB beta; // auxiliary variables for
    int binary_i[2], binary_j[2], binary_m[2];
    int k, L, h;
    for (int i = 0; i < number_of_states; i++) {
        integer_to_binary(i, binary_i);
        for (int j = 0; j < number_of_states; j++) {
            integer_to_binary(j, binary_j);
            for (int m = 0; m < number_of_states; m++) {
                integer_to_binary(m, binary_m);
                for (int index = 0; index < number_of_basis; index++) {
                    k = basis[index][0];
                    L = basis[index][1];
                    h = basis[index][2];
                    beta[i][j][m][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                    obj_func_1 += beta[i][j][m][index] * weight[i][j][m][index];
                    curr_RHS = (state[2 * k] - binary_i[0]) * (2 * binary_i[0] - 1);
                    curr_RHS += (state[2 * k + 1] - binary_i[1]) * (2 * binary_i[1] - 1);
                    curr_RHS += (state[2 * L] - binary_j[0]) * (2 * binary_j[0] - 1);
                    curr_RHS += (state[2 * L + 1] - binary_j[1]) * (2 * binary_j[1] - 1);
                    curr_RHS += (state[2 * h] - binary_m[0]) * (2 * binary_m[0] - 1);
                    curr_RHS += (state[2 * h + 1] - binary_m[1]) * (2 * binary_m[1] - 1);
                    model.addConstr(beta[i][j][m][index] >= 1 + curr_RHS);
                    model.addConstr(beta[i][j][m][index] <= (2 * binary_i[0] - 1) * state[2 * k] + 1 - binary_i[0]);
                    model.addConstr(beta[i][j][m][index] <= (2 * binary_i[1] - 1) * state[2 * k + 1] + 1 - binary_i[1]);
                    model.addConstr(beta[i][j][m][index] <= (2 * binary_j[0] - 1) * state[2 * L] + 1 - binary_j[0]);
                    model.addConstr(beta[i][j][m][index] <= (2 * binary_j[1] - 1) * state[2 * L + 1] + 1 - binary_j[1]);
                    model.addConstr(beta[i][j][m][index] <= (2 * binary_m[0] - 1) * state[2 * h] + 1 - binary_m[0]);
                    model.addConstr(beta[i][j][m][index] <= (2 * binary_m[1] - 1) * state[2 * h + 1] + 1 - binary_m[1]);
                }
            }
        }
    }

    typedef GRBVar NumVar8D_GRB[number_of_states][number_of_states][number_of_states][number_of_states][number_of_states][2][2][2][number_of_basis];
    NumVar8D_GRB xi;     // auxiliary variables for g
    int binary_i1[2], binary_j1[2], binary_i2[2], binary_j2[2], binary_h1[2], binary_j3[2];
    for (int i1 = 0; i1 < number_of_states; i1++) {
        integer_to_binary(i1, binary_i1);
        for (int j1 = 0; j1 < number_of_states; j1++) {
            integer_to_binary(j1, binary_j1);
            for (int h1 = 0; h1 < number_of_states; h1++) {
                integer_to_binary(h1, binary_h1);
                for (int i2 = 0; i2 < number_of_states; i2++) {
                    integer_to_binary(i2, binary_i2);
                    for (int j2 = 0; j2 < number_of_states; j2++) {
                        integer_to_binary(j2, binary_j2);
                        for (int ai = 0; ai < 2; ai++) {
                            for (int aj = 0; aj < 2; aj++) {
                                for (int ah = 0; ah < 2; ah++) {
                                    for (int index = 0; index < number_of_basis; index++) {
                                        k = basis[index][0];
                                        L = basis[index][1];
                                        h = basis[index][2];
                                        xi[i1][j1][h1][i2][j2][ai][aj][ah][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_i1[0] - 1) * state[2 * k] + 1 - binary_i1[0]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_i1[1] - 1) * state[2 * k + 1] + 1 - binary_i1[1]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_j1[0] - 1) * state[2 * L] + 1 - binary_j1[0]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_j1[1] - 1) * state[2 * L + 1] + 1 - binary_j1[1]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_h1[0] - 1) * state[2 * h] + 1 - binary_h1[0]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_h1[1] - 1) * state[2 * h + 1] + 1 - binary_h1[1]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_i2[0] - 1) * state[2 * topology_basis_scope_three[index][0]] + 1 - binary_i2[0]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_i2[1] - 1) * state[2 * topology_basis_scope_three[index][0] + 1] + 1 - binary_i2[1]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_j2[0] - 1) * state[2 * topology_basis_scope_three[index][1]] + 1 - binary_j2[0]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * binary_j2[1] - 1) * state[2 * topology_basis_scope_three[index][1] + 1] + 1 - binary_j2[1]);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * ai - 1) * action[k] + 1 - ai);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * aj - 1) * action[L] + 1 - aj);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] <= (2 * ah - 1) * action[h] + 1 - ah);

                                        curr_RHS = (2 * binary_i1[0] - 1) * (state[2 * k] - binary_i1[0]);
                                        curr_RHS += (2 * binary_i1[1] - 1) * (state[2 * k + 1] - binary_i1[1]);
                                        curr_RHS += (2 * binary_j1[0] - 1) * (state[2 * L] - binary_j1[0]);
                                        curr_RHS += (2 * binary_j1[1] - 1) * (state[2 * L + 1] - binary_j1[1]);
                                        curr_RHS += (2 * binary_h1[0] - 1) * (state[2 * h] - binary_h1[0]);
                                        curr_RHS += (2 * binary_h1[1] - 1) * (state[2 * h + 1] - binary_h1[1]);
                                        curr_RHS += (2 * binary_i2[0] - 1) * (state[2 * topology_basis_scope_three[index][0]] - binary_i2[0]);
                                        curr_RHS += (2 * binary_i2[1] - 1) * (state[2 * topology_basis_scope_three[index][0] + 1] - binary_i2[1]);
                                        curr_RHS += (2 * binary_j2[0] - 1) * (state[2 * topology_basis_scope_three[index][1]] - binary_j2[0]);
                                        curr_RHS += (2 * binary_j2[1] - 1) * (state[2 * topology_basis_scope_three[index][1] + 1] - binary_j2[1]);
                                        curr_RHS += (2 * ai - 1) * (action[k] - ai);
                                        curr_RHS += (2 * aj - 1) * (action[L] - aj);
                                        curr_RHS += (2 * ah - 1) * (action[h] - ah);
                                        model.addConstr(xi[i1][j1][h1][i2][j2][ai][aj][ah][index] >= 1 + curr_RHS);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double prob_i, prob_j, prob_h;
    int index_k1, index_k2;
    for (int i0 = 0; i0 < number_of_states; i0++) {
        for (int j0 = 0; j0 < number_of_states; j0++) {
            for (int h0 = 0; h0 < number_of_states; h0++) {
                for (int i1 = 0; i1 < number_of_states; i1++) {
                    for (int j1 = 0; j1 < number_of_states; j1++) {
                        for (int h1 = 0; h1 < number_of_states; h1++) {
                            for (int i2 = 0; i2 < number_of_states; i2++) {
                                for (int j2 = 0; j2 < number_of_states; j2++) {
                                    for (int ai = 0; ai < 2; ai++) {
                                        for (int aj = 0; aj < 2; aj++) {
                                            for (int ah = 0; ah < 2; ah++) {
                                                for (int index = 0; index < number_of_basis; index++) {
                                                    k = basis[index][0];
                                                    L = basis[index][1];
                                                    h = basis[index][2];
                                                    if (ai == 1) {
                                                        if (i0 == number_of_states - 1) prob_i = 1.0;
                                                        else prob_i = 0.0;
                                                    }
                                                    else {
                                                        if (neighbors[k][0] == k) index_k1 = i1;
                                                        if (neighbors[k][0] == L) index_k1 = j1;
                                                        if (neighbors[k][0] == h) index_k1 = h1;
                                                        if (neighbors[k][0] == topology_basis_scope_three[index][0]) index_k1 = i2;
                                                        if (neighbors[k][0] == topology_basis_scope_three[index][1]) index_k1 = j2;
                                                        if (neighbors[k][1] == k) index_k2 = i1;
                                                        if (neighbors[k][1] == L) index_k2 = j1;
                                                        if (neighbors[k][1] == h) index_k2 = h1;
                                                        if (neighbors[k][1] == topology_basis_scope_three[index][0]) index_k2 = i2;
                                                        if (neighbors[k][1] == topology_basis_scope_three[index][1]) index_k2 = j2;
                                                        prob_i = prob[i0][i1][index_k1][index_k2];
                                                    }
                                                    if (aj == 1) {
                                                        if (j0 == number_of_states - 1) prob_j = 1.0;
                                                        else prob_j = 0.0;
                                                    }
                                                    else {
                                                        if (neighbors[L][0] == k) index_k1 = i1;
                                                        if (neighbors[L][0] == L) index_k1 = j1;
                                                        if (neighbors[L][0] == h) index_k1 = h1;
                                                        if (neighbors[L][0] == topology_basis_scope_three[index][0]) index_k1 = i2;
                                                        if (neighbors[L][0] == topology_basis_scope_three[index][1]) index_k1 = j2;
                                                        if (neighbors[L][1] == k) index_k2 = i1;
                                                        if (neighbors[L][1] == L) index_k2 = j1;
                                                        if (neighbors[L][1] == h) index_k2 = h1;
                                                        if (neighbors[L][1] == topology_basis_scope_three[index][0]) index_k2 = i2;
                                                        if (neighbors[L][1] == topology_basis_scope_three[index][1]) index_k2 = j2;
                                                        prob_j = prob[j0][j1][index_k1][index_k2];
                                                    }

                                                    if (ah == 1) {
                                                        if (h0 == number_of_states - 1) prob_h = 1.0;
                                                        else prob_h = 0.0;
                                                    }
                                                    else {
                                                        if (neighbors[h][0] == k) index_k1 = i1;
                                                        if (neighbors[h][0] == L) index_k1 = j1;
                                                        if (neighbors[h][0] == h) index_k1 = h1;
                                                        if (neighbors[h][0] == topology_basis_scope_three[index][0]) index_k1 = i2;
                                                        if (neighbors[h][0] == topology_basis_scope_three[index][1]) index_k1 = j2;
                                                        if (neighbors[h][1] == k) index_k2 = i1;
                                                        if (neighbors[h][1] == L) index_k2 = j1;
                                                        if (neighbors[h][1] == h) index_k2 = h1;
                                                        if (neighbors[h][1] == topology_basis_scope_three[index][0]) index_k2 = i2;
                                                        if (neighbors[h][1] == topology_basis_scope_three[index][1]) index_k2 = j2;
                                                        prob_h = prob[h0][h1][index_k1][index_k2];
                                                    }

                                                    obj_func_2 += prob_i * prob_j * prob_h * weight[i0][j0][h0][index] * xi[i1][j1][h1][i2][j2][ai][aj][ah][index];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    GRBLinExpr obj = obj_func_1 - discount_factor * obj_func_2;
    model.setObjective(obj, GRB_MINIMIZE);

    // generate the constraints: x0[k] + x1[k] + x2[k] = 1 for all k
    GRBLinExpr sum_action;
    for (int k = 0; k < number_of_computers; ++k) sum_action += action[k];
    model.addConstr(sum_action == number_of_actions);

    for (int k = 0; k < number_of_computers; ++k) model.addConstr(state[2 * k] + 2 * state[2 * k + 1] <= number_of_states - 1);

    // solve the problem
    model.set(GRB_IntParam_Threads, number_of_Threads);
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Cuts, 1);
    env->set(GRB_DoubleParam_Cutoff, eps);

    // 进行热启动
    model.getEnv().set(GRB_IntParam_Presolve, 0);
    MyCallback cb(eps, time_limit_least);
    model.setCallback(&cb);

    model.getEnv().set(GRB_DoubleParam_TimeLimit, time_limit_most);

    model.optimize();

    double objValue = model.get(GRB_DoubleAttr_ObjVal);
    double bestBound = model.get(GRB_DoubleAttr_ObjBound);
    double gap = model.get(GRB_DoubleAttr_MIPGap);
    // std::cout << "Objective function value: " << objValue << std::endl;
    // std::cout << "Best bound: " << bestBound << std::endl;
    // std::cout << "Gap: " << gap << std::endl;

    // cout << "*** SLAVE OBJECTIVE VALUE: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    // cout << "sub problem tackled in : " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    // add constraint to master problem if necessary
    for (int i = 0; i < number_of_computers; i++) {
        worst_state[i] = state[2 * i].get(GRB_DoubleAttr_X) + 2 * state[2 * i + 1].get(GRB_DoubleAttr_X);
        if (action[i].get(GRB_DoubleAttr_X) > 0.5) worst_action[i] = 1;
        else worst_action[i] = 0;
    }

    // that's it!
    double value = model.get(GRB_DoubleAttr_ObjVal);

    delete env;

    return value;
}
