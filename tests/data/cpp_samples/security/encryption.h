/**
 * @file encryption.h
 * @brief Security encryption interface
 */

#pragma once
#include <string>
#include <vector>
#include <memory>

namespace Security {

/**
 * @brief Encryption algorithms supported
 */
enum class EncryptionAlgorithm {
    AES_128,
    AES_256,
    RSA,
    BLOWFISH,
    CUSTOM
};

/**
 * @brief Base interface for encryption operations
 */
class Encryptor {
public:
    /**
     * @requirement REQ-SEC-001
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~Encryptor() = default;
    
    /**
     * @requirement REQ-SEC-002
     * @brief Encrypts the provided plaintext
     * @param plaintext Data to encrypt
     * @return Encrypted data
     */
    virtual std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext) = 0;
    
    /**
     * @requirement REQ-SEC-003
     * @brief Decrypts the provided ciphertext
     * @param ciphertext Data to decrypt
     * @return Decrypted data
     */
    virtual std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext) = 0;
    
    /**
     * @requirement REQ-SEC-004
     * @brief Gets the algorithm being used
     * @return Encryption algorithm
     */
    virtual EncryptionAlgorithm getAlgorithm() const = 0;
    
    /**
     * @requirement REQ-SEC-005
     * @brief Sets the encryption key
     * @param key Key data
     * @return True if key is valid and was set, false otherwise
     */
    virtual bool setKey(const std::vector<uint8_t>& key) = 0;
    
    /**
     * @requirement REQ-SEC-006
     * @brief Checks if the encryptor is properly initialized
     * @return True if initialized, false otherwise
     */
    virtual bool isInitialized() const = 0;
};

/**
 * @brief Helper function to convert a string to a vector of bytes
 * @param str String to convert
 * @return Byte vector
 */
std::vector<uint8_t> stringToBytes(const std::string& str);

/**
 * @brief Helper function to convert a vector of bytes to a string
 * @param bytes Byte vector to convert
 * @return String representation
 */
std::string bytesToString(const std::vector<uint8_t>& bytes);

/**
 * @requirement REQ-SEC-007
 * @brief Factory function to create an encryptor for the specified algorithm
 * @param algorithm Encryption algorithm to use
 * @return Unique pointer to encryptor
 */
std::unique_ptr<Encryptor> createEncryptor(EncryptionAlgorithm algorithm);

/**
 * @requirement REQ-SEC-008
 * @brief Generates a random key suitable for the specified algorithm
 * @param algorithm Algorithm to generate key for
 * @return Generated key bytes
 */
std::vector<uint8_t> generateKey(EncryptionAlgorithm algorithm);

/**
 * @requirement REQ-SEC-009
 * @brief Validates that a key is suitable for the specified algorithm
 * @param algorithm Algorithm to validate against
 * @param key Key to validate
 * @return True if key is valid, false otherwise
 */
bool validateKey(EncryptionAlgorithm algorithm, const std::vector<uint8_t>& key);

} // namespace Security
