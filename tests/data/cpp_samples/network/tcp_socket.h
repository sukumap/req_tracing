/**
 * @file tcp_socket.h
 * @brief TCP socket implementation
 */

#pragma once
#include "socket.h"

namespace Network {

/**
 * @brief TCP socket implementation
 */
class TCPSocket : public Socket {
public:
    /**
     * @requirement REQ-NET-010
     * @brief Constructor for a TCP socket
     * @param host Target host
     * @param port Target port
     */
    TCPSocket(const std::string& host, int port);
    
    /**
     * @requirement REQ-NET-011
     * @brief Destructor for TCP socket
     */
    ~TCPSocket() override;
    
    /**
     * @requirement REQ-NET-012
     * @brief Connects to the remote TCP host
     * @return True if connection is successful, false otherwise
     */
    bool connect() override;
    
    /**
     * @requirement REQ-NET-013
     * @brief Disconnects from the remote TCP host
     * @return True if disconnection is successful, false otherwise
     */
    bool disconnect() override;
    
    /**
     * @requirement REQ-NET-014
     * @brief Sends data over the TCP socket
     * @param data Data to send
     * @return Number of bytes sent, -1 on error
     */
    int send(const std::vector<uint8_t>& data) override;
    
    /**
     * @requirement REQ-NET-015
     * @brief Receives data from the TCP socket
     * @param maxBytes Maximum number of bytes to receive
     * @return Data received, empty vector on error
     */
    std::vector<uint8_t> receive(size_t maxBytes) override;
    
    /**
     * @requirement REQ-NET-016
     * @brief Sets TCP-specific socket options
     * @param noDelay If true, disables Nagle's algorithm
     * @return True if option was set successfully, false otherwise
     */
    bool setTcpNoDelay(bool noDelay);
    
    /**
     * @requirement REQ-NET-017
     * @brief Listens for incoming connections
     * @param backlog Maximum number of queued connections
     * @return True if successful, false otherwise
     */
    bool listen(int backlog = 10);
    
    /**
     * @requirement REQ-NET-018
     * @brief Accepts an incoming connection
     * @return New socket for the accepted connection, nullptr on error
     */
    std::unique_ptr<TCPSocket> accept();

private:
    bool noDelay_;
    int backlog_;
};

} // namespace Network
