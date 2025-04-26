/**
 * @file tcp_socket.cpp
 * @brief TCP socket implementation
 */

#include "tcp_socket.h"
#include <iostream>
#include <thread>
#include <chrono>

namespace Network {

TCPSocket::TCPSocket(const std::string& host, int port)
    : Socket(host, port), noDelay_(false), backlog_(10) {
    std::cout << "TCP Socket created for " << host << ":" << port << std::endl;
}

TCPSocket::~TCPSocket() {
    std::cout << "TCP Socket destroyed" << std::endl;
}

bool TCPSocket::connect() {
    if (state_ == SocketState::CONNECTED) {
        std::cout << "Already connected" << std::endl;
        return true;
    }
    
    std::cout << "Connecting to " << host_ << ":" << port_ << "..." << std::endl;
    state_ = SocketState::CONNECTING;
    
    // Simulate connection process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // In a real implementation, this would use platform-specific socket APIs
    socketHandle_ = 12345; // Dummy socket handle
    state_ = SocketState::CONNECTED;
    
    std::cout << "Connected successfully" << std::endl;
    return true;
}

bool TCPSocket::disconnect() {
    if (state_ != SocketState::CONNECTED) {
        std::cout << "Not connected" << std::endl;
        return false;
    }
    
    std::cout << "Disconnecting from " << host_ << ":" << port_ << "..." << std::endl;
    
    // Simulate disconnection process
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // In a real implementation, this would use platform-specific socket APIs
    socketHandle_ = -1;
    state_ = SocketState::DISCONNECTED;
    
    std::cout << "Disconnected successfully" << std::endl;
    return true;
}

int TCPSocket::send(const std::vector<uint8_t>& data) {
    if (state_ != SocketState::CONNECTED) {
        std::cerr << "Cannot send: not connected" << std::endl;
        return -1;
    }
    
    if (data.empty()) {
        std::cout << "No data to send" << std::endl;
        return 0;
    }
    
    std::cout << "Sending " << data.size() << " bytes..." << std::endl;
    
    // Simulate sending data
    std::this_thread::sleep_for(std::chrono::milliseconds(data.size() / 100));
    
    // In a real implementation, this would use platform-specific socket APIs
    std::cout << "Sent " << data.size() << " bytes successfully" << std::endl;
    
    return static_cast<int>(data.size());
}

std::vector<uint8_t> TCPSocket::receive(size_t maxBytes) {
    if (state_ != SocketState::CONNECTED) {
        std::cerr << "Cannot receive: not connected" << std::endl;
        return {};
    }
    
    if (maxBytes == 0) {
        std::cout << "Requested to receive 0 bytes" << std::endl;
        return {};
    }
    
    std::cout << "Receiving up to " << maxBytes << " bytes..." << std::endl;
    
    // Simulate receiving data (in a real implementation, this would be actual received data)
    size_t bytesReceived = maxBytes / 2 + (maxBytes % 2); // Simulate receiving about half the requested bytes
    
    // Simulate receive delay
    std::this_thread::sleep_for(std::chrono::milliseconds(bytesReceived / 50));
    
    // Create dummy received data
    std::vector<uint8_t> receivedData(bytesReceived, 0x42); // Fill with dummy value 0x42
    
    std::cout << "Received " << receivedData.size() << " bytes" << std::endl;
    return receivedData;
}

bool TCPSocket::setTcpNoDelay(bool noDelay) {
    std::cout << (noDelay ? "Enabling" : "Disabling") << " TCP_NODELAY option" << std::endl;
    
    // In a real implementation, this would set the TCP_NODELAY socket option
    noDelay_ = noDelay;
    
    std::cout << "TCP_NODELAY option " << (noDelay ? "enabled" : "disabled") << std::endl;
    return true;
}

bool TCPSocket::listen(int backlog) {
    if (state_ == SocketState::LISTENING) {
        std::cout << "Already listening" << std::endl;
        return true;
    }
    
    if (state_ == SocketState::CONNECTED) {
        std::cout << "Cannot listen: already connected as a client" << std::endl;
        return false;
    }
    
    std::cout << "Starting to listen on " << host_ << ":" << port_ 
              << " with backlog " << backlog << std::endl;
    
    // In a real implementation, this would use platform-specific socket APIs
    backlog_ = backlog;
    state_ = SocketState::LISTENING;
    
    std::cout << "Listening successfully" << std::endl;
    return true;
}

std::unique_ptr<TCPSocket> TCPSocket::accept() {
    if (state_ != SocketState::LISTENING) {
        std::cerr << "Cannot accept: not listening" << std::endl;
        return nullptr;
    }
    
    std::cout << "Waiting for incoming connection..." << std::endl;
    
    // Simulate connection acceptance (in reality, this would block until a connection arrives)
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Create a new socket for the accepted connection
    auto clientSocket = std::make_unique<TCPSocket>("client", 12345);
    clientSocket->state_ = SocketState::CONNECTED;
    clientSocket->socketHandle_ = 54321; // Dummy handle for accepted socket
    
    std::cout << "Accepted new connection" << std::endl;
    return clientSocket;
}

} // namespace Network
