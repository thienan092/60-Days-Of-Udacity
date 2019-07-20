#pragma once
#include "definitions.h"

class Features
{
	void calc_AtoB(Object** Player_A, Object** Player_B);
	void calc_Bmap();

public:
	int AtoB[N_PLAYER][N_PLAYER];
	int Bmap[N_PLAYER];

	static void states_to_complex_states(Object& oBall, Object** Player_A, Object** Player_B, float* state, int* Bmap);

	Features(Object** Player_A, Object** Player_B, Object& oBall);
	~Features();
};

