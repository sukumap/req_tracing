/**
 * @file utils.cpp
 * @brief Implementation of utility functions
 */

#include "utils.h"
#include <algorithm>

/**
 * @requirement REQ-UTIL-001
 * Implementation of string concatenation
 */
std::string concat(const std::string& a, const std::string& b) {
    return a + b;
}

/**
 * @requirement REQ-UTIL-002
 * Implementation of uppercase conversion
 */
std::string toUpper(const std::string& input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}
