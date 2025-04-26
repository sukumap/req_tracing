/**
 * @file socket.cpp
 * @brief Network socket implementation
 */

#include "socket.h"
#include <stdexcept>
#include <iostream>

namespace Network {

Socket::Socket() : host_("localhost"), port_(8080), state_(SocketState::DISCONNECTED), socketHandle_(-1) {
    std::cout << "Socket created with default parameters" << std::endl;
}

Socket::Socket(const std::string& host, int port) 
    : host_(host), port_(port), state_(SocketState::DISCONNECTED), socketHandle_(-1) {
    std::cout << "Socket created for " << host << ":" << port << std::endl;
}

Socket::~Socket() {
    if (state_ == SocketState::CONNECTED) {
        std::cout << "Warning: Socket destroyed while still connected" << std::endl;
        disconnect();
    }
    
    if (socketHandle_ != -1) {
        // Close socket handle
        std::cout << "Closing socket handle: " << socketHandle_ << std::endl;
        socketHandle_ = -1;
    }
}

SocketState Socket::getState() const {
    return state_;
}

bool Socket::setOption(const std::string& optionName, const std::string& optionValue) {
    std::cout << "Setting socket option " << optionName << " to " << optionValue << std::endl;
    
    // Implementation would typically call platform-specific socket option functions
    if (optionName.empty() || optionValue.empty()) {
        return false;
    }
    
    // Simulate setting options
    if (optionName == "timeout") {
        try {
            int timeout = std::stoi(optionValue);
            std::cout << "Socket timeout set to " << timeout << " ms" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Invalid timeout value: " << e.what() << std::endl;
            return false;
        }
    } else if (optionName == "keepalive") {
        bool enabled = (optionValue == "true" || optionValue == "1");
        std::cout << "Socket keepalive " << (enabled ? "enabled" : "disabled") << std::endl;
        return true;
    }
    
    std::cerr << "Unknown socket option: " << optionName << std::endl;
    return false;
}

// Factory function implementation
std::unique_ptr<Socket> createSocket(const std::string& type, 
                                   const std::string& host, 
                                   int port) {
    // In a real implementation, this would return different socket subclasses
    // based on the type parameter (TCP, UDP, etc.)
    std::cout << "Creating " << type << " socket for " << host << ":" << port << std::endl;
    
    // This is a simplified example - in reality we would create and return
    // concrete instances of different socket types
    if (type == "tcp" || type == "udp" || type == "ssl") {
        std::cout << "Socket type " << type << " created" << std::endl;
        // For now, we'll return nullptr as we're just providing examples
        return nullptr;
    }
    
    throw std::invalid_argument("Unsupported socket type: " + type);
}

} // namespace Network
