/**
 * @file utils.h
 * @brief Utility functions
 */

#ifndef UTILS_H
#define UTILS_H

#include <string>

/**
 * @requirement REQ-UTIL-001
 * @brief String concatenation utility
 * @param a First string
 * @param b Second string
 * @return Concatenated result
 */
std::string concat(const std::string& a, const std::string& b);

/**
 * @requirement REQ-UTIL-002
 * @brief Convert string to uppercase
 * @param input Input string
 * @return Uppercase version of input
 */
std::string toUpper(const std::string& input);

#endif // UTILS_H
