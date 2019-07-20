#pragma once
#include <math.h>
#include "definitions.h"

void ChangePos(Object& A, int myTeamID, int match)
{
	if (((myTeamID == 0) && (match == HALF_1)) || ((myTeamID == 1) && (match != HALF_1)))
		return;
	A.m_pos.x = mapWidth - A.m_pos.x;
	A.m_pos.y = mapHeight - A.m_pos.y;
	A.m_moveSpeed.x = -A.m_moveSpeed.x;
	A.m_moveSpeed.y = -A.m_moveSpeed.y;
}

void ChangePos(int& x, int&y, int myTeamID, int match)
{
	if (((myTeamID == 0) && (match == HALF_1)) || ((myTeamID == 1) && (match != HALF_1)))
		return;
	x = mapWidth - x;
	y = mapHeight - y;
}

bool CanShoot(Position& mPos, Position& ballPos)
{
	if (RAG_TO_SHOOT >= DistancePos(mPos, ballPos))
		return true;
	return false;
}

float DistancePos(Position& PosA, Position& PosB) {
	float dis = sqrt((PosA.x - PosB.x)*(PosA.x - PosB.x) + (PosA.y - PosB.y)*(PosA.y - PosB.y));
	return dis;
}

void to_pos_x_y(Object& player, int q_id, int& x, int& y)
{
	x = q_id / H_BINS;
	y = q_id % H_BINS;
	x = (int)(x * (((float)mapWidth) / (W_BINS - 1)));
	y = (int)(y * (((float)mapHeight) / (H_BINS - 1)));
}

void to_pos_x_y_f(Object& player, int q_id, int& x, int& y, int& f)
{
	x = q_id / (H_BINS * FORCE_FD);
	int r = q_id % (H_BINS * FORCE_FD);
	y = r / FORCE_FD;
	f = r % FORCE_FD;
	x = (int)(x * (((float)mapWidth) / (W_BINS - 1)));
	y = (int)(y * (((float)mapHeight) / (H_BINS - 1)));
	f = (int)(f * (((float)(FORCE_MAX - FORCE_MIN)) / (FORCE_FD - 1))) + FORCE_MIN;
}