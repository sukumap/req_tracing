/**
 * @file encryption.cpp
 * @brief Implementation of encryption functionality
 */

#include "encryption.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>

namespace Security {

// Helper functions implementation
std::vector<uint8_t> stringToBytes(const std::string& str) {
    return std::vector<uint8_t>(str.begin(), str.end());
}

std::string bytesToString(const std::vector<uint8_t>& bytes) {
    return std::string(bytes.begin(), bytes.end());
}

// Internal implementation classes
namespace {
    // Base class for encryption implementations
    class BaseEncryptor : public Encryptor {
    protected:
        bool initialized_ = false;
        std::vector<uint8_t> key_;
        EncryptionAlgorithm algorithm_;
        
    public:
        explicit BaseEncryptor(EncryptionAlgorithm algo) : algorithm_(algo) {}
        
        bool isInitialized() const override {
            return initialized_;
        }
        
        EncryptionAlgorithm getAlgorithm() const override {
            return algorithm_;
        }
    };
    
    /**
     * @requirement REQ-SEC-010
     * @brief AES encryption implementation
     */
    class AESEncryptor : public BaseEncryptor {
    private:
        bool is256Bit_;
        
    public:
        explicit AESEncryptor(bool use256Bit) 
            : BaseEncryptor(use256Bit ? EncryptionAlgorithm::AES_256 : EncryptionAlgorithm::AES_128),
              is256Bit_(use256Bit) {
            std::cout << "Created AES-" << (use256Bit ? "256" : "128") << " encryptor" << std::endl;
        }
        
        bool setKey(const std::vector<uint8_t>& key) override {
            // Check key length
            size_t requiredKeySize = is256Bit_ ? 32 : 16;  // 256 bits = 32 bytes, 128 bits = 16 bytes
            if (key.size() != requiredKeySize) {
                std::cerr << "Invalid key size for AES-" << (is256Bit_ ? "256" : "128") 
                          << ". Expected " << requiredKeySize << " bytes, got " 
                          << key.size() << " bytes." << std::endl;
                return false;
            }
            
            key_ = key;
            initialized_ = true;
            std::cout << "AES key set successfully" << std::endl;
            return true;
        }
        
        std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext) override {
            if (!initialized_) {
                throw std::runtime_error("AES encryptor not initialized with a key");
            }
            
            std::cout << "Encrypting " << plaintext.size() << " bytes with AES-" 
                      << (is256Bit_ ? "256" : "128") << std::endl;
            
            // In a real implementation, this would use a cryptography library
            // This is just a placeholder that performs a simple XOR with the key
            std::vector<uint8_t> ciphertext(plaintext);
            for (size_t i = 0; i < ciphertext.size(); ++i) {
                ciphertext[i] ^= key_[i % key_.size()];
            }
            
            return ciphertext;
        }
        
        std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext) override {
            if (!initialized_) {
                throw std::runtime_error("AES decryptor not initialized with a key");
            }
            
            std::cout << "Decrypting " << ciphertext.size() << " bytes with AES-" 
                      << (is256Bit_ ? "256" : "128") << std::endl;
            
            // For simple XOR, encryption and decryption are the same operation
            return encrypt(ciphertext);
        }
    };
    
    /**
     * @requirement REQ-SEC-011
     * @brief RSA encryption implementation
     */
    class RSAEncryptor : public BaseEncryptor {
    private:
        std::vector<uint8_t> publicKey_;
        std::vector<uint8_t> privateKey_;
        
    public:
        RSAEncryptor() : BaseEncryptor(EncryptionAlgorithm::RSA) {
            std::cout << "Created RSA encryptor" << std::endl;
        }
        
        bool setKey(const std::vector<uint8_t>& key) override {
            // In a real implementation, this would parse a proper RSA key
            // For demonstration, we'll just split the key in half
            if (key.size() < 64) {  // Arbitrary minimum size
                std::cerr << "RSA key too small, minimum 64 bytes required" << std::endl;
                return false;
            }
            
            size_t halfSize = key.size() / 2;
            publicKey_.assign(key.begin(), key.begin() + halfSize);
            privateKey_.assign(key.begin() + halfSize, key.end());
            initialized_ = true;
            std::cout << "RSA key set successfully" << std::endl;
            return true;
        }
        
        std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext) override {
            if (!initialized_) {
                throw std::runtime_error("RSA encryptor not initialized with keys");
            }
            
            std::cout << "Encrypting " << plaintext.size() << " bytes with RSA" << std::endl;
            
            // In a real implementation, this would use RSA encryption
            // This is just a placeholder for demonstration
            std::vector<uint8_t> ciphertext;
            ciphertext.reserve(plaintext.size() + 8);  // Add room for a header
            
            // Add a mock header
            ciphertext.push_back('R');
            ciphertext.push_back('S');
            ciphertext.push_back('A');
            ciphertext.push_back('E');
            ciphertext.push_back(0x01);
            ciphertext.push_back(0x02);
            ciphertext.push_back(0x03);
            ciphertext.push_back(0x04);
            
            // Append "encrypted" data (in reality, just a simple transform)
            for (uint8_t byte : plaintext) {
                ciphertext.push_back(~byte);  // Bitwise NOT as simple transform
            }
            
            return ciphertext;
        }
        
        std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext) override {
            if (!initialized_) {
                throw std::runtime_error("RSA decryptor not initialized with keys");
            }
            
            std::cout << "Decrypting " << ciphertext.size() << " bytes with RSA" << std::endl;
            
            // Check for minimal size and header
            if (ciphertext.size() < 8 || 
                ciphertext[0] != 'R' || 
                ciphertext[1] != 'S' || 
                ciphertext[2] != 'A' || 
                ciphertext[3] != 'E') {
                throw std::invalid_argument("Invalid RSA ciphertext format");
            }
            
            // Extract the payload and "decrypt" it
            std::vector<uint8_t> plaintext;
            plaintext.reserve(ciphertext.size() - 8);
            
            for (size_t i = 8; i < ciphertext.size(); ++i) {
                plaintext.push_back(~ciphertext[i]);  // Reverse the bitwise NOT
            }
            
            return plaintext;
        }
    };
}

// Public function implementations
std::unique_ptr<Encryptor> createEncryptor(EncryptionAlgorithm algorithm) {
    switch (algorithm) {
        case EncryptionAlgorithm::AES_128:
            return std::make_unique<AESEncryptor>(false);
        case EncryptionAlgorithm::AES_256:
            return std::make_unique<AESEncryptor>(true);
        case EncryptionAlgorithm::RSA:
            return std::make_unique<RSAEncryptor>();
        case EncryptionAlgorithm::BLOWFISH:
            std::cout << "Blowfish encryption not implemented yet" << std::endl;
            return nullptr;
        case EncryptionAlgorithm::CUSTOM:
            std::cout << "Custom encryption requires specific implementation" << std::endl;
            return nullptr;
        default:
            std::cerr << "Unknown encryption algorithm" << std::endl;
            return nullptr;
    }
}

std::vector<uint8_t> generateKey(EncryptionAlgorithm algorithm) {
    // Generate a random key of appropriate length for the algorithm
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    
    size_t keySize = 0;
    switch (algorithm) {
        case EncryptionAlgorithm::AES_128:
            keySize = 16;  // 128 bits = 16 bytes
            break;
        case EncryptionAlgorithm::AES_256:
            keySize = 32;  // 256 bits = 32 bytes
            break;
        case EncryptionAlgorithm::RSA:
            keySize = 128;  // Just a placeholder size for demonstration
            break;
        case EncryptionAlgorithm::BLOWFISH:
            keySize = 56;  // Maximum key size for Blowfish
            break;
        default:
            throw std::invalid_argument("Unknown or unsupported algorithm for key generation");
    }
    
    std::vector<uint8_t> key(keySize);
    std::generate(key.begin(), key.end(), [&]() { return dist(gen); });
    
    std::cout << "Generated " << keySize << " byte key for " 
              << static_cast<int>(algorithm) << " algorithm" << std::endl;
    
    return key;
}

bool validateKey(EncryptionAlgorithm algorithm, const std::vector<uint8_t>& key) {
    // Check if the key meets the requirements for the algorithm
    switch (algorithm) {
        case EncryptionAlgorithm::AES_128:
            return key.size() == 16;
        case EncryptionAlgorithm::AES_256:
            return key.size() == 32;
        case EncryptionAlgorithm::RSA:
            return key.size() >= 64;  // Arbitrary minimum size
        case EncryptionAlgorithm::BLOWFISH:
            return key.size() > 0 && key.size() <= 56;
        default:
            return false;
    }
}

} // namespace Security
