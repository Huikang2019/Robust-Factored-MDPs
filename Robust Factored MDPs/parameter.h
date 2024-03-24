
#ifndef PARAMETER_H
#define PARAMETER_H
#include <ilcplex/ilocplex.h>
#include <gurobi_c++.h>
#include <list>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <time.h>
#include <chrono>
#include <thread>
#include <atomic>

using namespace std;

#define ILOUSESTL

const int       number_of_computers = 20;
const int       number_of_rings = number_of_computers / 3;
const int       number_of_half = number_of_computers / 2;
const int       number_of_actions = 2;
const double    discount_factor = 0.95;
const int       number_of_states = 3;
const int		number_of_constraints = 30000;
const double    lambda = 0.0;
const double    correlation = 0.0;

const int       number_of_Threads = 8;
const int       job_number = 1;
const int       scope = 2;
const int       number_of_neighbors = 2;
const double    bigM = 100000;

const int       number_of_state_actions = number_of_states * number_of_states * number_of_states * 2;

typedef GRBVar NumVar2D[number_of_states][number_of_computers];
typedef GRBVar NumVar2DA[2][number_of_computers];
typedef double NumArray2D[number_of_states][number_of_computers];
typedef double NumArray2DA[2][number_of_computers];
typedef double NumArray3D[number_of_states][number_of_states][number_of_computers];
typedef GRBVar NumVar6D[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers][number_of_computers];
typedef GRBVar NumVar3D[number_of_states][number_of_states][number_of_computers];
typedef GRBVar NumVar8D[number_of_states][number_of_states][number_of_states][number_of_states][2][2][number_of_computers];
typedef GRBVar NumVarAction[2][2][number_of_computers];


#endif
#pragma once

