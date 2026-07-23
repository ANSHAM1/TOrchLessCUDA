#pragma once

#include <cstddef>

void tensorAssign(float* ptr, std::size_t value, std::size_t n);

void tensorAssignRandom(float* data, std::size_t size, unsigned long long seed);