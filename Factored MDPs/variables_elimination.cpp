#include "parameter.h"
#include "function.h"

extern int          neighbors[number_of_computers][number_of_neighbors];
extern double       prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern int          basis[number_of_basis][scope];
extern NumArray3D   weight3;
extern NumArray2D   weight2;


void solve_overall_problem_scope_1_VE() {
    // set up Gurobi model
    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_OutputFlag, 0);
    GRBModel model = GRBModel(env);


    NumVar2D alpha;
    GRBQuadExpr obj_func = 0.0;
    for (int i = 0; i < number_of_states; i++) {
        for (int k = 0; k < number_of_basis; k++) {
            alpha[i][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
            obj_func += alpha[i][k] / number_of_states;
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.update();

    // generate the constraints (one for each state)

    typedef GRBVar NumVar55D[number_of_states][number_of_states][number_of_states][number_of_computers][number_of_basis];

    NumVar55D beta;
    double prob_i, prob_j;
    GRBLinExpr curr_RHS = 0.0;
    for (int t = 0; t < number_of_states; t++) {
        for (int i = 0; i < number_of_states; i++) {
            for (int j = 0; j < number_of_states; j++) {
                for (int action = 0; action < number_of_computers; action++) {   // denote the action
                    for (int l = 0; l < number_of_basis; l++) {   // denote the state index
                        beta[t][i][j][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                        curr_RHS = 0.0;
                        for (int state_i = 0; state_i < number_of_states; ++state_i) {
                            if (action == l) {
                                if (state_i == number_of_states - 1) prob_i = 1.0;
                                else prob_i = 0.0;
                            }
                            else prob_i = prob[state_i][i][t][j];
                            curr_RHS += prob_i * alpha[state_i][l];
                        }
                        model.addConstr(beta[t][i][j][action][l] == discount_factor * curr_RHS);
                    }
                }

            }
        }
    }


    typedef GRBVar NumVar66D[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers][number_of_basis - 4];
    NumVar66D zeta;
    for (int t2 = 0; t2 < number_of_states; t2++) {
        for (int t1 = 0; t1 < number_of_states; t1++) {
            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int action = 0; action < number_of_computers; action++) {   // denote the action
                        zeta[t2][t1][i][j][action][0] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                        for (int t0 = 0; t0 < number_of_states; ++t0) {
                            curr_RHS = t0 - alpha[t0][0];
                            curr_RHS += beta[t2][t1][t0][action][number_of_computers - 1];
                            curr_RHS += beta[t1][t0][i][action][0];
                            curr_RHS += beta[t0][i][j][action][1];
                            model.addConstr(zeta[t2][t1][i][j][action][0] >= curr_RHS);
                        }
                    }
                }

            }
        }
    }

    for (int l = 1; l < number_of_basis - 4; l++) {   // denote the state index
        for (int t2 = 0; t2 < number_of_states; t2++) {
            for (int t1 = 0; t1 < number_of_states; t1++) {
                for (int i = 0; i < number_of_states; i++) {
                    for (int j = 0; j < number_of_states; j++) {
                        for (int action = 0; action < number_of_computers; action++) {   // denote the action
                            zeta[t2][t1][i][j][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                            for (int t0 = 0; t0 < number_of_states; ++t0) {
                                curr_RHS = t0 - alpha[t0][l] + beta[t0][i][j][action][l + 1];
                                curr_RHS += zeta[t2][t1][t0][i][action][l - 1];
                                model.addConstr(zeta[t2][t1][i][j][action][l] >= curr_RHS);
                            }
                        }
                    }
                }
            }
        }
    }

    int l = number_of_basis - 4;
    for (int t2 = 0; t2 < number_of_states; t2++) {
        for (int t1 = 0; t1 < number_of_states; t1++) {
            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int action = 0; action < number_of_computers; action++) {   // denote the action                        
                        curr_RHS = t2 + t1 + i + j;
                        curr_RHS -= alpha[i][l] + alpha[j][l + 1] + alpha[t2][l + 2] + alpha[t1][l + 3];
                        curr_RHS += beta[i][j][t2][action][l + 1] + beta[j][t2][t1][action][l + 2];
                        curr_RHS += zeta[t2][t1][i][j][action][l - 1];
                        model.addConstr(0 >= curr_RHS);
                    }
                }
            }
        }
    }
    // solve the problem

    //model.set(GRB_IntParam_OutputFlag, 0);
    model.optimize();


    cout << "objective value of order " << 1 << " is: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    cout << "problem tackled in " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int k = 0; k < number_of_basis; k++) {
            weight2[i][k] = alpha[i][k].get(GRB_DoubleAttr_X);
        }
    }
}

void solve_overall_problem_scope_1_VE_two_actions() {

    // set up Gurobi model
    GRBEnv* env = new GRBEnv();
    env->set(GRB_IntParam_OutputFlag, 0);
    GRBModel model = GRBModel(env);
        
    try {
        NumVar2D alpha;
        GRBQuadExpr obj_func = 0.0;
        for (int i = 0; i < number_of_states; i++) {
            for (int k = 0; k < number_of_basis; k++) {
                alpha[i][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                obj_func += alpha[i][k] / number_of_states;
            }
        }

        model.setObjective(obj_func, GRB_MINIMIZE);
        model.update();

        // generate the constraints (one for each state)

        typedef GRBVar NumVar55D[number_of_states][number_of_states][number_of_states][number_of_computers][number_of_computers][number_of_basis];

        
        NumVar55D beta;
        double prob_i, prob_j;
        GRBLinExpr curr_RHS = 0.0;
        for (int t = 0; t < number_of_states; t++) {
            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int action1 = 0; action1 < number_of_computers - 1; action1++) {   // denote the action
                        for (int action2 = action1 + 1; action2 < number_of_computers; action2++) {
                            for (int l = 0; l < number_of_basis; l++) {   // denote the state index
                                beta[t][i][j][action1][action2][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                                curr_RHS = 0.0;
                                for (int state_i = 0; state_i < number_of_states; ++state_i) {
                                    if ((action1 == l) || (action2 == l)) {
                                        if (state_i == number_of_states - 1) prob_i = 1.0;
                                        else prob_i = 0.0;
                                    }
                                    else prob_i = prob[state_i][i][t][j];
                                    curr_RHS += prob_i * alpha[state_i][l];
                                }
                                model.addConstr(beta[t][i][j][action1][action2][l] == discount_factor * curr_RHS);
                            }
                        }
                    }

                }
            }
        }


        typedef GRBVar NumVar66D[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers][number_of_computers][number_of_basis - 4];
        NumVar66D zeta;
        for (int t2 = 0; t2 < number_of_states; t2++) {
            for (int t1 = 0; t1 < number_of_states; t1++) {
                for (int i = 0; i < number_of_states; i++) {
                    for (int j = 0; j < number_of_states; j++) {
                        for (int action1 = 0; action1 < number_of_computers - 1; action1++) {   // denote the action
                            for (int action2 = action1 + 1; action2 < number_of_computers; action2++) {   // denote the action
                                zeta[t2][t1][i][j][action1][action2][0] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                                for (int t0 = 0; t0 < number_of_states; ++t0) {
                                    curr_RHS = t0 - alpha[t0][0];
                                    curr_RHS += beta[t2][t1][t0][action1][action2][number_of_computers - 1];
                                    curr_RHS += beta[t1][t0][i][action1][action2][0];
                                    curr_RHS += beta[t0][i][j][action1][action2][1];
                                    model.addConstr(zeta[t2][t1][i][j][action1][action2][0] >= curr_RHS);
                                }
                            }

                        }
                    }

                }
            }
        }

        for (int l = 1; l < number_of_basis - 4; l++) {   // denote the state index
            for (int t2 = 0; t2 < number_of_states; t2++) {
                for (int t1 = 0; t1 < number_of_states; t1++) {
                    for (int i = 0; i < number_of_states; i++) {
                        for (int j = 0; j < number_of_states; j++) {
                            for (int action1 = 0; action1 < number_of_computers - 1; action1++) {   // denote the action
                                for (int action2 = action1 + 1; action2 < number_of_computers; action2++) {
                                    zeta[t2][t1][i][j][action1][action2][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                                    for (int t0 = 0; t0 < number_of_states; ++t0) {
                                        curr_RHS = t0 - alpha[t0][l] + beta[t0][i][j][action1][action2][l + 1];
                                        curr_RHS += zeta[t2][t1][t0][i][action1][action2][l - 1];
                                        model.addConstr(zeta[t2][t1][i][j][action1][action2][l] >= curr_RHS);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        int l = number_of_basis - 4;
        for (int t2 = 0; t2 < number_of_states; t2++) {
            for (int t1 = 0; t1 < number_of_states; t1++) {
                for (int i = 0; i < number_of_states; i++) {
                    for (int j = 0; j < number_of_states; j++) {
                        for (int action1 = 0; action1 < number_of_computers - 1; action1++) {   // denote the action 
                            for (int action2 = action1 + 1; action2 < number_of_computers; action2++) {
                                curr_RHS = t2 + t1 + i + j;
                                curr_RHS -= alpha[i][l] + alpha[j][l + 1] + alpha[t2][l + 2] + alpha[t1][l + 3];
                                curr_RHS += beta[i][j][t2][action1][action2][l + 1] + beta[j][t2][t1][action1][action2][l + 2];
                                curr_RHS += zeta[t2][t1][i][j][action1][action2][l - 1];
                                model.addConstr(0 >= curr_RHS);
                            }
                        }
                    }
                }
            }
        }
        // solve the problem

        //model.set(GRB_IntParam_OutputFlag, 0);
        model.optimize();


        cout << "objective value of order " << 1 << " is: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
        cout << "problem tackled in " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

        for (int i = 0; i < number_of_states; i++) {
            for (int k = 0; k < number_of_basis; k++) {
                weight2[i][k] = alpha[i][k].get(GRB_DoubleAttr_X);
            }
        }
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        delete env;
    }
        
    
}

void solve_overall_problem_scope_2_VE() {
    // set up Gurobi model
    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_OutputFlag, 0);
    GRBModel model = GRBModel(env);

    NumVar3D alpha;
    GRBQuadExpr obj_func = 0.0;
    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_basis; k++) {
                alpha[i][j][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                obj_func += alpha[i][j][k] / pow(number_of_states, scope);
            }
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.update();

    // generate the constraints (one for each state)
    typedef GRBVar NumVar66D[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers][number_of_basis];
    NumVar66D beta;
    double prob_i, prob_j;
    GRBLinExpr curr_RHS = 0.0;
    for (int t = 0; t < number_of_states; t++) {
        for (int i = 0; i < number_of_states; i++) {
            for (int j = 0; j < number_of_states; j++) {
                for (int k = 0; k < number_of_states; k++) {
                    for (int action = 0; action < number_of_computers; action++) {   // denote the action
                        for (int l = 0; l < number_of_basis; l++) {   // denote the state index
                            beta[t][i][j][k][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                            curr_RHS = 0.0;
                            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                                for (int state_j = 0; state_j < number_of_states; ++state_j) {
                                    if (action == l) {
                                        if (state_i == number_of_states - 1) prob_i = 1.0;
                                        else prob_i = 0.0;
                                    }
                                    else prob_i = prob[state_i][i][t][j];
                                    if (action == incr(l)) {
                                        if (state_j == number_of_states - 1) prob_j = 1.0;
                                        else prob_j = 0.0;
                                    }
                                    else prob_j = prob[state_j][j][k][i];
                                    curr_RHS += prob_i * prob_j * alpha[state_i][state_j][l];
                                }
                            }
                            model.addConstr(beta[t][i][j][k][action][l] == discount_factor * curr_RHS);
                        }
                    }
                }
            }
        }
    }

    typedef GRBVar NumVar88D[number_of_states][number_of_states][number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers][number_of_basis - 6];
    NumVar88D zeta;
    for (int t3 = 0; t3 < number_of_states; t3++) {
        for (int t2 = 0; t2 < number_of_states; t2++) {
            for (int t1 = 0; t1 < number_of_states; t1++) {
                for (int i = 0; i < number_of_states; i++) {
                    for (int j = 0; j < number_of_states; j++) {
                        for (int k = 0; k < number_of_states; k++) {
                            for (int action = 0; action < number_of_computers; action++) {   // denote the action
                                zeta[t3][t2][t1][i][j][k][action][0] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                                for (int t0 = 0; t0 < number_of_states; ++t0) {
                                    curr_RHS = t0;
                                    curr_RHS -= alpha[t1][t0][number_of_computers - 1] + alpha[t0][i][0];
                                    curr_RHS += beta[t3][t2][t1][t0][action][number_of_computers - 2];
                                    curr_RHS += beta[t2][t1][t0][i][action][number_of_computers - 1];
                                    curr_RHS += beta[t1][t0][i][j][action][0];
                                    curr_RHS += beta[t0][i][j][k][action][1];
                                    model.addConstr(zeta[t3][t2][t1][i][j][k][action][0] >= curr_RHS);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int l = 1; l < number_of_basis - 6; l++) {
        for (int t3 = 0; t3 < number_of_states; t3++) {
            for (int t2 = 0; t2 < number_of_states; t2++) {
                for (int t1 = 0; t1 < number_of_states; t1++) {
                    for (int i = 0; i < number_of_states; i++) {
                        for (int j = 0; j < number_of_states; j++) {
                            for (int k = 0; k < number_of_states; k++) {
                                for (int action = 0; action < number_of_computers; action++) {   // denote the action                                  
                                    // denote the state index
                                    zeta[t3][t2][t1][i][j][k][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                                    for (int state_i = 0; state_i < number_of_states; ++state_i) {
                                        curr_RHS = state_i - alpha[state_i][i][l] + beta[state_i][i][j][k][action][l + 1];
                                        curr_RHS += zeta[t3][t2][t1][state_i][i][j][action][l - 1];
                                        model.addConstr(zeta[t3][t2][t1][i][j][k][action][l] >= curr_RHS);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    int l = number_of_basis - 6;
    for (int t3 = 0; t3 < number_of_states; t3++) {
        for (int t2 = 0; t2 < number_of_states; t2++) {
            for (int t1 = 0; t1 < number_of_states; t1++) {
                for (int i = 0; i < number_of_states; i++) {
                    for (int j = 0; j < number_of_states; j++) {
                        for (int k = 0; k < number_of_states; k++) {
                            for (int action = 0; action < number_of_computers; action++) {   // denote the action                                                       
                                curr_RHS = t3 + t2 + t1 + i + j + k;
                                curr_RHS -= alpha[i][j][l] + alpha[j][k][l + 1] + alpha[k][t3][l + 2] + alpha[t3][t2][l + 3] + alpha[t2][t1][l + 4];
                                curr_RHS += beta[i][j][k][t3][action][l + 1] + beta[j][k][t3][t2][action][l + 2] + beta[k][t3][t2][t1][action][l + 3];
                                curr_RHS += zeta[t3][t2][t1][i][j][k][action][l - 1];
                                model.addConstr(0 >= curr_RHS);
                            }
                        }
                    }
                }
            }
        }
    }
    // solve the problem
    // env.set("OutputFlag", 0);
    // model.set(GRB_IntParam_LogToConsole, 0);
    model.set(GRB_IntParam_OutputFlag, 1);
    model.optimize();


    cout << "objective value of order " << 2 << " is: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    cout << "problem tackled in " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_basis; k++) {
                weight3[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
            }
        }
    }
}

void solve_overall_problem_scope_2_VE_two_actions() {
    // set up Gurobi model
    GRBEnv* env = new GRBEnv();
    env->set(GRB_IntParam_OutputFlag, 0);
    GRBModel model = GRBModel(env);

    try {
        NumVar3D alpha;
        GRBQuadExpr obj_func = 0.0;
        for (int i = 0; i < number_of_states; i++) {
            for (int j = 0; j < number_of_states; j++) {
                for (int k = 0; k < number_of_basis; k++) {
                    alpha[i][j][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                    obj_func += alpha[i][j][k] / pow(number_of_states, scope);
                }
            }
        }

        model.setObjective(obj_func, GRB_MINIMIZE);
        model.update();

        // generate the constraints (one for each state)
        typedef GRBVar NumVar66D[number_of_states][number_of_states][number_of_states][number_of_states]
            [number_of_computers][number_of_computers][number_of_basis];
        NumVar66D beta;
        double prob_i, prob_j;
        GRBLinExpr curr_RHS = 0.0;
        for (int t = 0; t < number_of_states; t++) {
            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int k = 0; k < number_of_states; k++) {
                        for (int action = 0; action < number_of_computers - 1; action++) {   // denote the action
                            for (int action2 = action + 1; action2 < number_of_computers; action2++) {
                                for (int l = 0; l < number_of_basis; l++) {   // denote the state index
                                    beta[t][i][j][k][action][action2][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                                    curr_RHS = 0.0;
                                    for (int state_i = 0; state_i < number_of_states; ++state_i) {
                                        for (int state_j = 0; state_j < number_of_states; ++state_j) {
                                            if ((action == l) || (action2 == l)) {
                                                if (state_i == number_of_states - 1) prob_i = 1.0;
                                                else prob_i = 0.0;
                                            }
                                            else prob_i = prob[state_i][i][t][j];
                                            if ((action == incr(l)) || (action2 == incr(l))) {
                                                if (state_j == number_of_states - 1) prob_j = 1.0;
                                                else prob_j = 0.0;
                                            }
                                            else prob_j = prob[state_j][j][k][i];
                                            curr_RHS += prob_i * prob_j * alpha[state_i][state_j][l];
                                        }
                                    }
                                    model.addConstr(beta[t][i][j][k][action][action2][l] == discount_factor * curr_RHS);
                                }
                            }

                        }
                    }
                }
            }
        }

        typedef GRBVar NumVar88D[number_of_states][number_of_states][number_of_states][number_of_states][number_of_states]
            [number_of_states][number_of_computers][number_of_computers][number_of_basis - 6];
        NumVar88D zeta;
        for (int t3 = 0; t3 < number_of_states; t3++) {
            for (int t2 = 0; t2 < number_of_states; t2++) {
                for (int t1 = 0; t1 < number_of_states; t1++) {
                    for (int i = 0; i < number_of_states; i++) {
                        for (int j = 0; j < number_of_states; j++) {
                            for (int k = 0; k < number_of_states; k++) {
                                for (int action = 0; action < number_of_computers - 1; action++) {   // denote the action
                                    for (int action2 = action + 1; action2 < number_of_computers; action2++) {
                                        zeta[t3][t2][t1][i][j][k][action][action2][0] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                                        for (int t0 = 0; t0 < number_of_states; ++t0) {
                                            curr_RHS = t0;
                                            curr_RHS -= alpha[t1][t0][number_of_computers - 1] + alpha[t0][i][0];
                                            curr_RHS += beta[t3][t2][t1][t0][action][action2][number_of_computers - 2];
                                            curr_RHS += beta[t2][t1][t0][i][action][action2][number_of_computers - 1];
                                            curr_RHS += beta[t1][t0][i][j][action][action2][0];
                                            curr_RHS += beta[t0][i][j][k][action][action2][1];
                                            model.addConstr(zeta[t3][t2][t1][i][j][k][action][action2][0] >= curr_RHS);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (int l = 1; l < number_of_basis - 6; l++) {
            for (int t3 = 0; t3 < number_of_states; t3++) {
                for (int t2 = 0; t2 < number_of_states; t2++) {
                    for (int t1 = 0; t1 < number_of_states; t1++) {
                        for (int i = 0; i < number_of_states; i++) {
                            for (int j = 0; j < number_of_states; j++) {
                                for (int k = 0; k < number_of_states; k++) {
                                    for (int action = 0; action < number_of_computers; action++) {   // denote the action  
                                        for (int action2 = action + 1; action2 < number_of_computers; action2++) {
                                            // denote the state index
                                            zeta[t3][t2][t1][i][j][k][action][action2][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                                            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                                                curr_RHS = state_i - alpha[state_i][i][l] + beta[state_i][i][j][k][action][action2][l + 1];
                                                curr_RHS += zeta[t3][t2][t1][state_i][i][j][action][action2][l - 1];
                                                model.addConstr(zeta[t3][t2][t1][i][j][k][action][action2][l] >= curr_RHS);
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

        int l = number_of_basis - 6;
        for (int t3 = 0; t3 < number_of_states; t3++) {
            for (int t2 = 0; t2 < number_of_states; t2++) {
                for (int t1 = 0; t1 < number_of_states; t1++) {
                    for (int i = 0; i < number_of_states; i++) {
                        for (int j = 0; j < number_of_states; j++) {
                            for (int k = 0; k < number_of_states; k++) {
                                for (int action = 0; action < number_of_computers; action++) {   // denote the action 
                                    for (int action2 = action + 1; action2 < number_of_computers; action2++) {
                                        curr_RHS = t3 + t2 + t1 + i + j + k;
                                        curr_RHS -= alpha[i][j][l] + alpha[j][k][l + 1] + alpha[k][t3][l + 2] + alpha[t3][t2][l + 3] + alpha[t2][t1][l + 4];
                                        curr_RHS += beta[i][j][k][t3][action][action2][l + 1] + beta[j][k][t3][t2][action][action2][l + 2] + beta[k][t3][t2][t1][action][action2][l + 3];
                                        curr_RHS += zeta[t3][t2][t1][i][j][k][action][action2][l - 1];
                                        model.addConstr(0 >= curr_RHS);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // solve the problem
        // env.set("OutputFlag", 0);
        // model.set(GRB_IntParam_LogToConsole, 0);
        model.set(GRB_IntParam_OutputFlag, 0);
        model.optimize();


        cout << "objective value of order " << 2 << " is: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
        cout << "problem tackled in " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

        for (int i = 0; i < number_of_states; i++) {
            for (int j = 0; j < number_of_states; j++) {
                for (int k = 0; k < number_of_basis; k++) {
                    weight3[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                }
            }
        }

    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        delete env;
    }
}

void solve_overall_problem_scope_1_VE_RoR() {
    // set up Gurobi model
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    NumVar2D alpha;
    GRBQuadExpr obj_func = 0.0;
    for (int i = 0; i < number_of_states; i++) {
        for (int k = 0; k < number_of_basis; k++) {
            alpha[i][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
            obj_func += alpha[i][k] / number_of_states;
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.update();

    // generate the constraints (one for each state)


    typedef GRBVar NumVar44D[number_of_states][number_of_states][number_of_computers][number_of_rings];

    NumVar44D beta0, beta1, beta2, beta3;
    double prob_i, prob_j;
    GRBLinExpr curr_RHS = 0.0;
    for (int t = 0; t < number_of_states; t++) {
        for (int i = 0; i < number_of_states; i++) {
            for (int action = 0; action < number_of_computers; action++) {   // denote the action
                for (int l = 0; l < number_of_rings; l++) {   // denote the state index
                    beta0[t][i][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                    curr_RHS = 0.0;
                    for (int state_i = 0; state_i < number_of_states; ++state_i) {
                        if (action == 3 * l) {
                            if (state_i == number_of_states - 1) prob_i = 1.0;
                            else prob_i = 0.0;
                        }
                        else prob_i = prob[state_i][i][t][i];
                        curr_RHS += prob_i * alpha[state_i][3 * l];
                    }
                    model.addConstr(beta0[t][i][action][l] == discount_factor * curr_RHS);

                    beta1[t][i][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                    curr_RHS = 0.0;
                    for (int state_i = 0; state_i < number_of_states; ++state_i) {
                        if (action == 3 * l + 1) {
                            if (state_i == number_of_states - 1) prob_i = 1.0;
                            else prob_i = 0.0;
                        }
                        else prob_i = prob[state_i][i][t][i];
                        curr_RHS += prob_i * alpha[state_i][3 * l + 1];
                    }
                    model.addConstr(beta1[t][i][action][l] == discount_factor * curr_RHS);

                    beta2[t][i][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                    for (int j = 0; j < number_of_states; j++) {
                        curr_RHS = 0.0;
                        for (int state_i = 0; state_i < number_of_states; ++state_i) {
                            if (action == 3 * l + 2) {
                                if (state_i == number_of_states - 1) prob_i = 1.0;
                                else prob_i = 0.0;
                            }
                            else prob_i = prob[state_i][j][t][i];
                            curr_RHS += prob_i * alpha[state_i][3 * l + 2];
                        }
                        model.addConstr(beta2[t][i][action][l] >= discount_factor * curr_RHS + j - alpha[j][3 * l + 2]);
                    }

                }

            }

        }
    }



    typedef GRBVar NumVar33D[number_of_states][number_of_computers][number_of_rings];
    NumVar33D zeta;
    for (int i = 0; i < number_of_states; i++) {
        for (int action = 0; action < number_of_computers; action++) {
            for (int l = 0; l < number_of_rings; l++) {
                zeta[i][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                for (int state_i = 0; state_i < number_of_states; state_i++) {
                    model.addConstr(zeta[i][action][l] >= beta1[i][state_i][action][l] + beta2[i][state_i][action][l] + state_i - alpha[state_i][3 * l + 1]);
                }
            }

        }
    }

    for (int t = 0; t < number_of_states; t++) {
        for (int i = 0; i < number_of_states; i++) {
            for (int action = 0; action < number_of_computers; action++) {   // denote the action
                beta3[t][i][action][0] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                for (int state_i = 0; state_i < number_of_states; state_i++) {
                    curr_RHS = beta0[t][state_i][action][0] + beta0[state_i][i][action][number_of_rings - 1];
                    curr_RHS += zeta[state_i][action][0] + state_i - alpha[state_i][0];
                    model.addConstr(beta3[t][i][action][0] >= curr_RHS);
                }
            }
        }
    }


    for (int l = 1; l < number_of_rings - 2; l++) {
        for (int t = 0; t < number_of_states; t++) {
            for (int i = 0; i < number_of_states; i++) {
                for (int action = 0; action < number_of_computers; action++) {
                    // denote the state index

                    beta3[t][i][action][l] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                    for (int state_i = 0; state_i < number_of_states; state_i++) {
                        curr_RHS = beta0[t][state_i][action][l] + beta3[t][state_i][action][l - 1];
                        curr_RHS += zeta[state_i][action][l] + state_i - alpha[state_i][3 * l];
                        model.addConstr(beta3[t][i][action][l] >= curr_RHS);
                    }

                }
            }
        }
    }



    for (int t = 0; t < number_of_states; t++) {
        for (int i = 0; i < number_of_states; i++) {
            for (int action = 0; action < number_of_computers; action++) {
                int l = number_of_rings - 2;
                curr_RHS = beta0[t][i][action][l] + beta3[t][i][action][l - 1];
                curr_RHS += zeta[t][action][l] + t - alpha[t][3 * l];
                curr_RHS += zeta[i][action][l + 1] + i - alpha[i][3 * (l + 1)];
                model.addConstr(0 >= curr_RHS);

            }

        }
    }

    // solve the problem
    model.set(GRB_IntParam_OutputFlag, 0);
    model.optimize();


    cout << "objective value of order " << 1 << " is: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    cout << "problem tackled in " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int k = 0; k < number_of_basis; k++) {
            weight2[i][k] = alpha[i][k].get(GRB_DoubleAttr_X);
        }
    }
}

