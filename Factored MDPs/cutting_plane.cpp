#include "parameter.h"
#include "function.h"

double cutting_plane_scope_one(NumArray2D& weight) {

    GRBEnv* env = new GRBEnv();

    GRBModel model = GRBModel(env);

    NumVar2D alpha;

    creat_mastered_problem_scope_one(model, alpha);

    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Threads, number_of_Threads);
    model.set(GRB_IntParam_Method, GRB_METHOD_DUAL);

    double mastertime = 0;

    int state[number_of_computers], action[number_of_computers];
    int state0, action0;
    double const_violation;

    double value = 0, slackValue;
    int number_of_constraints, ratio = 10, curr_iter = 0;
    for (int iter = 0; iter < 10 * number_of_computers; iter++) {
        for (int inner_iter = 0; inner_iter < 6; inner_iter++) {
            random_generate_constraints_local_search_scope_one(model, alpha, weight, 0);
            curr_iter += number_of_computers;
            if (curr_iter > 2 * number_of_computers) {
                model.optimize();
                mastertime += model.get(GRB_DoubleAttr_Runtime);
                number_of_constraints = model.get(GRB_IntAttr_NumConstrs);
                if (number_of_constraints > ratio * model.get(GRB_IntAttr_NumVars)) {
                    GRBConstr* NBen = model.getConstrs();
                    for (int i = 0; i < model.get(GRB_IntAttr_NumConstrs); i++) {
                        slackValue = NBen[i].get(GRB_DoubleAttr_Slack);
                        // Use the slackValue as needed (e.g., check if it's large enough and delete the constraint)
                        if ((slackValue >= 0.001 * value) && (slackValue >= 0.1)) {
                            model.remove(NBen[i]);
                            number_of_constraints--;
                        }
                    }
                    if (number_of_constraints > (ratio - 3) * model.get(GRB_IntAttr_NumVars)) {
                        ratio += 5;
                    }
                    delete[] NBen;
                }

                curr_iter = 0;

                for (int i = 0; i < number_of_states; i++) {
                    for (int k = 0; k < number_of_basis; k++) {
                        weight[i][k] = alpha[i][k].get(GRB_DoubleAttr_X);
                    }
                }
            }

        }

        // cout << "number of constraints:" << model.get(GRB_IntAttr_NumConstrs) << "  " << endl;
        // cout << "value of master problem: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
        
        if ((model.get(GRB_DoubleAttr_ObjVal) < (1 + eps) * value) && (iter > 5 * number_of_computers)) {
            const_violation = solve_subed_problem_GRB_scope_one(weight, -10 * eps * value, 10, 100 + 3 * number_of_computers, state, action);
            //iterations_of_sub++;
            cout << "constraint violation is: " << const_violation << endl << endl;
            fixed_constraints_local_search_scope_one(model, alpha, weight, state, action);
            model.optimize();
            mastertime += model.get(GRB_DoubleAttr_Runtime);
            for (int i = 0; i < number_of_states; i++) {
                for (int k = 0; k < number_of_basis; k++) {
                    weight[i][k] = alpha[i][k].get(GRB_DoubleAttr_X);
                }
            }
            if (const_violation > -10 * eps * value) break;
        }

        // cout << "master problem tackled in : " << mastertime << " seconds." << endl;

        value = model.get(GRB_DoubleAttr_ObjVal);

    }

    delete env;

    return value;
}

double cutting_plane_scope_two(NumArray3D& weight) {

    GRBEnv* env = new GRBEnv();

    GRBModel model = GRBModel(env);

    NumVar3D alpha;

    creat_mastered_problem_scope_two(model, alpha);

    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Threads, number_of_Threads);
    model.set(GRB_IntParam_Method, GRB_METHOD_DUAL);

    double mastertime = 0;

    int state[number_of_computers], action[number_of_computers];
    int state0, action0;
    double const_violation;

    double value = 0, slackValue;
    int number_of_constraints, ratio = 10, curr_iter = 0;
    for (int iter = 0; iter < 15 * number_of_computers; iter++) {
        for (int inner_iter = 0; inner_iter < 6; inner_iter++) {
            random_generate_constraints_local_search_scope_two(model, alpha, weight, 0);
            curr_iter += number_of_computers;
            if (curr_iter > 2 * number_of_computers) {
                model.optimize();
                mastertime += model.get(GRB_DoubleAttr_Runtime);
                number_of_constraints = model.get(GRB_IntAttr_NumConstrs);
                if (number_of_constraints > ratio * model.get(GRB_IntAttr_NumVars)) {
                    GRBConstr* NBen = model.getConstrs();
                    for (int i = 0; i < model.get(GRB_IntAttr_NumConstrs); i++) {
                        slackValue = NBen[i].get(GRB_DoubleAttr_Slack);
                        // Use the slackValue as needed (e.g., check if it's large enough and delete the constraint)
                        if ((slackValue >= 0.001 * value) && (slackValue >= 0.1)) {
                            model.remove(NBen[i]);
                            number_of_constraints--;
                        }
                    }
                    if (number_of_constraints > (ratio - 3) * model.get(GRB_IntAttr_NumVars)) {
                        ratio += 5;
                    }
                    delete[] NBen;
                }

                curr_iter = 0;

                for (int i = 0; i < number_of_states; i++) {
                    for (int j = 0; j < number_of_states; j++) {
                        for (int k = 0; k < number_of_basis; k++) {
                            weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                        }
                    }
                }
            }

        }

        // cout << "number of constraints:" << model.get(GRB_IntAttr_NumConstrs) << "  " << endl;
        // cout << "value of master problem: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

        if ((model.get(GRB_DoubleAttr_ObjVal) < (1 + eps) * value) && (iter > 5 * number_of_computers)) {
            const_violation = solve_subed_problem_GRB_scope_two(weight, -10 * eps * value, 10, 100 + 3 * number_of_computers, state, action);
            //iterations_of_sub++;
            fixed_constraints_local_search_scope_two(model, alpha, weight, state, action);
            model.optimize();
            mastertime += model.get(GRB_DoubleAttr_Runtime);
            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int k = 0; k < number_of_basis; k++) {
                        weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                    }
                }
            }
            if (const_violation > -10 * eps * value) break;
        }

        // cout << "master problem tackled in : " << mastertime << " seconds." << endl;

        value = model.get(GRB_DoubleAttr_ObjVal);

    }

    delete env;

    return value;
}

double cutting_plane_scope_three(NumArray4D& weight) {

    GRBEnv* env = new GRBEnv();

    GRBModel model = GRBModel(env);

    NumVar4D alpha;

    creat_mastered_problem_scope_three(model, alpha);

    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Threads, number_of_Threads);
    model.set(GRB_IntParam_Method, GRB_METHOD_DUAL);

    double mastertime = 0;

    int state[number_of_computers], action[number_of_computers];
    int state0, action0;
    double const_violation;

    double value = 0, slackValue;
    int number_of_constraints, ratio = 4, curr_iter = 0;
    for (int iter = 0; iter < 30 * number_of_computers; iter++) {
        for (int inner_iter = 0; inner_iter < 6; inner_iter++) {
            random_generate_constraints_local_search_scope_three(model, alpha, weight, 0);
            curr_iter += number_of_computers;
            if (curr_iter > 2 * number_of_computers) {
                model.optimize();
                mastertime += model.get(GRB_DoubleAttr_Runtime);
                number_of_constraints = model.get(GRB_IntAttr_NumConstrs);
                if (number_of_constraints > ratio * model.get(GRB_IntAttr_NumVars)) {
                    GRBConstr* NBen = model.getConstrs();
                    for (int i = 0; i < model.get(GRB_IntAttr_NumConstrs); i++) {
                        slackValue = NBen[i].get(GRB_DoubleAttr_Slack);
                        // Use the slackValue as needed (e.g., check if it's large enough and delete the constraint)
                        if ((slackValue >= 0.001 * value) && (slackValue >= 0.1)) {
                            model.remove(NBen[i]);
                            number_of_constraints--;
                        }
                    }
                    if (number_of_constraints > (ratio - 2) * model.get(GRB_IntAttr_NumVars)) {
                        ratio += 4;
                    }
                    delete[] NBen;
                }

                curr_iter = 0;

                for (int i = 0; i < number_of_states; i++) {
                    for (int j = 0; j < number_of_states; j++) {
                        for (int l = 0; l < number_of_states; l++) {
                            for (int k = 0; k < number_of_basis; k++) {
                                weight[i][j][l][k] = alpha[i][j][l][k].get(GRB_DoubleAttr_X);
                            }
                        }
                    }
                }
            }

        }

        // cout << "number of constraints:" << model.get(GRB_IntAttr_NumConstrs) << "  " << endl;
        // cout << "value of master problem: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

        if ((model.get(GRB_DoubleAttr_ObjVal) < (1 + eps) * value) && (iter > 5 * number_of_computers)) {
            const_violation = solve_subed_problem_GRB_scope_three(weight, -10 * eps * value, 10, 100 + 3 * number_of_computers, state, action);
            //iterations_of_sub++;
            fixed_constraints_local_search_scope_three(model, alpha, weight, state, action);
            model.optimize();
            mastertime += model.get(GRB_DoubleAttr_Runtime);
            for (int i = 0; i < number_of_states; i++) {
                for (int j = 0; j < number_of_states; j++) {
                    for (int l = 0; l < number_of_states; l++) {
                        for (int k = 0; k < number_of_basis; k++) {
                            weight[i][j][l][k] = alpha[i][j][l][k].get(GRB_DoubleAttr_X);
                        }
                    }
                }
            }
            if (const_violation > -10 * eps * value) break;
        }

        // cout << "master problem tackled in : " << mastertime << " seconds." << endl;

        value = model.get(GRB_DoubleAttr_ObjVal);

    }

    delete env;

    return value;
}
