/**
 * @file socket.h
 * @brief Network socket interface
 */

#pragma once
#include <string>
#include <vector>
#include <memory>

namespace Network {

/**
 * @brief Socket connection states
 */
enum class SocketState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    LISTENING,
    ERROR
};

/**
 * @brief Base socket class for network communications
 */
class Socket {
public:
    /**
     * @requirement REQ-NET-001
     * @brief Constructor initializes a socket with default parameters
     */
    Socket();
    
    /**
     * @requirement REQ-NET-001
     * @brief Constructor initializes a socket with specified parameters
     * @param host Host address
     * @param port Port number
     */
    Socket(const std::string& host, int port);
    
    /**
     * @requirement REQ-NET-002
     * @brief Destructor ensures proper socket cleanup
     */
    virtual ~Socket();
    
    /**
     * @requirement REQ-NET-003
     * @brief Connects to the remote host
     * @return True if connection is successful, false otherwise
     */
    virtual bool connect() = 0;
    
    /**
     * @requirement REQ-NET-004
     * @brief Disconnects from the remote host
     * @return True if disconnection is successful, false otherwise
     */
    virtual bool disconnect() = 0;
    
    /**
     * @requirement REQ-NET-005
     * @brief Sends data over the socket
     * @param data Data to send
     * @return Number of bytes sent, -1 on error
     */
    virtual int send(const std::vector<uint8_t>& data) = 0;
    
    /**
     * @requirement REQ-NET-006
     * @brief Receives data from the socket
     * @param maxBytes Maximum number of bytes to receive
     * @return Data received, empty vector on error
     */
    virtual std::vector<uint8_t> receive(size_t maxBytes) = 0;
    
    /**
     * @requirement REQ-NET-007
     * @brief Gets the current socket state
     * @return Current socket state
     */
    SocketState getState() const;
    
    /**
     * @requirement REQ-NET-008
     * @brief Sets socket options
     * @param optionName Name of the option to set
     * @param optionValue Value to set
     * @return True if option was set successfully, false otherwise
     */
    bool setOption(const std::string& optionName, const std::string& optionValue);

protected:
    std::string host_;
    int port_;
    SocketState state_;
    int socketHandle_;
};

/**
 * @requirement REQ-NET-009
 * @brief Create a socket of the specified type
 * @param type Type of socket to create ("tcp", "udp", etc.)
 * @param host Host address
 * @param port Port number
 * @return Unique pointer to created socket
 */
std::unique_ptr<Socket> createSocket(const std::string& type, 
                                    const std::string& host, 
                                    int port);

} // namespace Network
