/**
 * @file main.cpp
 * @brief Main application entry point
 */

#include "math.h"
#include "utils.h"
#include "core/app.h"
#include <iostream>
#include <string>

/**
 * @requirement REQ-MAIN-001
 * @brief Main function - entry point of the application
 * @return Exit code
 */
int main() {
    // Initialize the application
    if (!initializeApp("config.json")) {
        return 1;
    }

    // Demonstrate math functions
    int sum = add(5, 3);
    int difference = subtract(10, 4);
    int product = multiply(6, 7);
    
    std::cout << "Math results: " << sum << ", " << difference << ", " << product << std::endl;
    
    // Demonstrate string utilities
    std::string combined = concat("Hello, ", "World!");
    std::string uppercase = toUpper(combined);
    
    std::cout << "String results: " << combined << " -> " << uppercase << std::endl;
    
    // Process a command
    std::string result = processCommand("REPORT");
    std::cout << "Command result: " << result << std::endl;
    
    // Shutdown the application
    shutdownApp();
    
    return 0;
}
