/**
 * @file data_processor.h
 * @brief Data processing interfaces and utilities
 */

#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <map>

namespace Data {

/**
 * @brief Data format types supported by the processor
 */
enum class DataFormat {
    JSON,
    XML,
    CSV,
    BINARY,
    CUSTOM
};

/**
 * @brief Structure representing a data field
 */
struct DataField {
    std::string name;
    std::string value;
    std::string type;
};

/**
 * @brief Class representing a data record with multiple fields
 */
class DataRecord {
public:
    /**
     * @requirement REQ-DATA-001
     * @brief Creates an empty data record
     */
    DataRecord();
    
    /**
     * @requirement REQ-DATA-002
     * @brief Adds a field to the record
     * @param name Field name
     * @param value Field value
     * @param type Field type
     * @return Reference to this record for chaining
     */
    DataRecord& addField(const std::string& name, 
                         const std::string& value, 
                         const std::string& type = "string");
    
    /**
     * @requirement REQ-DATA-003
     * @brief Gets a field value by name
     * @param name Field name to retrieve
     * @return Field value or empty string if not found
     */
    std::string getFieldValue(const std::string& name) const;
    
    /**
     * @requirement REQ-DATA-004
     * @brief Gets all fields in the record
     * @return Vector of data fields
     */
    std::vector<DataField> getAllFields() const;
    
    /**
     * @requirement REQ-DATA-005
     * @brief Checks if a field exists in the record
     * @param name Field name to check
     * @return True if field exists, false otherwise
     */
    bool hasField(const std::string& name) const;
    
    /**
     * @requirement REQ-DATA-006
     * @brief Serializes the record to a string in the specified format
     * @param format Format to serialize to
     * @return Serialized record
     */
    std::string serialize(DataFormat format) const;

private:
    std::map<std::string, DataField> fields_;
};

/**
 * @brief Interface for data processors
 */
class DataProcessor {
public:
    /**
     * @requirement REQ-DATA-007
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~DataProcessor() = default;
    
    /**
     * @requirement REQ-DATA-008
     * @brief Processes a collection of data records
     * @param records Records to process
     * @return Processed result
     */
    virtual std::string process(const std::vector<DataRecord>& records) = 0;
    
    /**
     * @requirement REQ-DATA-009
     * @brief Validates a data record according to processor-specific rules
     * @param record Record to validate
     * @return True if valid, false otherwise
     */
    virtual bool validate(const DataRecord& record) = 0;
    
    /**
     * @requirement REQ-DATA-010
     * @brief Sets a configuration parameter for the processor
     * @param key Parameter key
     * @param value Parameter value
     */
    virtual void setConfig(const std::string& key, const std::string& value) = 0;
    
    /**
     * @requirement REQ-DATA-011
     * @brief Gets native format for this processor
     * @return Native data format
     */
    virtual DataFormat getNativeFormat() const = 0;
};

/**
 * @requirement REQ-DATA-012
 * @brief Factory function to create a data processor for the specified format
 * @param format Format the processor should handle
 * @return Unique pointer to a data processor
 */
std::unique_ptr<DataProcessor> createProcessor(DataFormat format);

/**
 * @requirement REQ-DATA-013
 * @brief Registers a data processor factory for a custom format
 * @param name Custom format name
 * @param factory Factory function to create the processor
 * @return True if registration succeeded, false otherwise
 */
bool registerCustomProcessor(
    const std::string& name,
    std::function<std::unique_ptr<DataProcessor>()> factory
);

} // namespace Data
