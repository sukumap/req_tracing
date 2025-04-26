/**
 * @file data_processor.cpp
 * @brief Data processing implementation
 */

#include "data_processor.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

namespace Data {

// DataRecord implementation
DataRecord::DataRecord() {
    std::cout << "Created empty data record" << std::endl;
}

DataRecord& DataRecord::addField(const std::string& name, 
                               const std::string& value, 
                               const std::string& type) {
    DataField field{name, value, type};
    fields_[name] = field;
    return *this;
}

std::string DataRecord::getFieldValue(const std::string& name) const {
    auto it = fields_.find(name);
    if (it != fields_.end()) {
        return it->second.value;
    }
    return "";
}

std::vector<DataField> DataRecord::getAllFields() const {
    std::vector<DataField> result;
    for (const auto& pair : fields_) {
        result.push_back(pair.second);
    }
    return result;
}

bool DataRecord::hasField(const std::string& name) const {
    return fields_.find(name) != fields_.end();
}

std::string DataRecord::serialize(DataFormat format) const {
    std::ostringstream oss;
    
    switch (format) {
        case DataFormat::JSON: {
            oss << "{";
            bool first = true;
            for (const auto& pair : fields_) {
                if (!first) oss << ",";
                oss << "\"" << pair.first << "\":";
                
                // Handle different types
                if (pair.second.type == "number") {
                    oss << pair.second.value;
                } else {
                    oss << "\"" << pair.second.value << "\"";
                }
                
                first = false;
            }
            oss << "}";
            break;
        }
        
        case DataFormat::XML: {
            oss << "<record>";
            for (const auto& pair : fields_) {
                oss << "<" << pair.first << " type=\"" << pair.second.type << "\">"
                    << pair.second.value
                    << "</" << pair.first << ">";
            }
            oss << "</record>";
            break;
        }
        
        case DataFormat::CSV: {
            // Headers
            bool first = true;
            for (const auto& pair : fields_) {
                if (!first) oss << ",";
                oss << pair.first;
                first = false;
            }
            oss << "\n";
            
            // Values
            first = true;
            for (const auto& pair : fields_) {
                if (!first) oss << ",";
                oss << pair.second.value;
                first = false;
            }
            break;
        }
        
        case DataFormat::BINARY:
            oss << "BINARY_DATA"; // Placeholder for binary serialization
            break;
            
        case DataFormat::CUSTOM:
            oss << "CUSTOM_FORMAT"; // Placeholder for custom format
            break;
    }
    
    return oss.str();
}

// Factory function implementations
namespace {
    std::unordered_map<std::string, std::function<std::unique_ptr<DataProcessor>()>> customProcessors;
    
    // Example processor implementations could be defined here
    class JsonProcessor : public DataProcessor {
    public:
        std::string process(const std::vector<DataRecord>& records) override {
            std::ostringstream oss;
            oss << "[";
            bool first = true;
            for (const auto& record : records) {
                if (!first) oss << ",";
                oss << record.serialize(DataFormat::JSON);
                first = false;
            }
            oss << "]";
            return oss.str();
        }
        
        bool validate(const DataRecord& record) override {
            // Simple validation - each record must have at least one field
            return !record.getAllFields().empty();
        }
        
        void setConfig(const std::string& key, const std::string& value) override {
            std::cout << "Setting JSON processor config: " << key << "=" << value << std::endl;
            // Implementation would store the configuration
        }
        
        DataFormat getNativeFormat() const override {
            return DataFormat::JSON;
        }
    };
    
    class XmlProcessor : public DataProcessor {
    public:
        std::string process(const std::vector<DataRecord>& records) override {
            std::ostringstream oss;
            oss << "<records>";
            for (const auto& record : records) {
                oss << record.serialize(DataFormat::XML);
            }
            oss << "</records>";
            return oss.str();
        }
        
        bool validate(const DataRecord& record) override {
            // Validate that all field names are valid XML element names
            // This is a simplified version - real XML validation would be more complex
            for (const auto& field : record.getAllFields()) {
                if (field.name.empty() || field.name[0] == ' ' || field.name.find(' ') != std::string::npos) {
                    return false;
                }
            }
            return true;
        }
        
        void setConfig(const std::string& key, const std::string& value) override {
            std::cout << "Setting XML processor config: " << key << "=" << value << std::endl;
            // Implementation would store the configuration
        }
        
        DataFormat getNativeFormat() const override {
            return DataFormat::XML;
        }
    };
}

std::unique_ptr<DataProcessor> createProcessor(DataFormat format) {
    switch (format) {
        case DataFormat::JSON:
            return std::make_unique<JsonProcessor>();
        case DataFormat::XML:
            return std::make_unique<XmlProcessor>();
        case DataFormat::CSV:
            std::cout << "CSV processor not implemented yet" << std::endl;
            return nullptr;
        case DataFormat::BINARY:
            std::cout << "Binary processor not implemented yet" << std::endl;
            return nullptr;
        case DataFormat::CUSTOM:
            std::cout << "Use registerCustomProcessor to define custom processors" << std::endl;
            return nullptr;
    }
    return nullptr;
}

bool registerCustomProcessor(
    const std::string& name,
    std::function<std::unique_ptr<DataProcessor>()> factory) {
    
    if (name.empty() || !factory) {
        return false;
    }
    
    auto result = customProcessors.insert({name, std::move(factory)});
    return result.second; // true if insertion happened
}

} // namespace Data
