/**
 * @file app.h
 * @brief Core application functionality
 */

#ifndef APP_H
#define APP_H

#include <string>

/**
 * @requirement REQ-APP-001
 * @brief Application initialization
 * @param configPath Path to configuration file
 * @return Success status
 */
bool initializeApp(const std::string& configPath);

/**
 * @requirement REQ-APP-002
 * @brief Application shutdown
 * @return Success status
 */
bool shutdownApp();

/**
 * @requirement REQ-APP-003
 * @brief Process command
 * @param command Command to process
 * @return Command result
 */
std::string processCommand(const std::string& command);

#endif // APP_H
