/**
 * @file math.h
 * @brief Math utility functions
 */

#ifndef MATH_H
#define MATH_H

/**
 * @requirement REQ-MATH-001
 * @brief Addition function
 * @param a First number to add
 * @param b Second number to add
 * @return Sum of a and b
 */
int add(int a, int b);

/**
 * @requirement REQ-MATH-002
 * @brief Subtraction function
 * @param a First number
 * @param b Number to subtract
 * @return Result of a - b
 */
int subtract(int a, int b);

/**
 * @requirement REQ-MATH-003
 * @brief Multiplication function
 * @param a First number
 * @param b Second number
 * @return Product of a and b
 */
int multiply(int a, int b);

#endif // MATH_H
