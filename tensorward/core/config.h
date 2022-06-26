#pragma once

#include <cassert>
#include <string_view>
#include <unordered_map>

namespace tensorward::core {

class Config {
 public:
  // Gets the singleton instance.
  static Config& instance() {
    static Config instance;
    return instance;
  }

  // Prevents copy construction.
  Config(const Config&) = delete;

  // Prevents move construction.
  Config(Config&&) = delete;

  // Prevents copy assignment.
  Config& operator=(const Config&) = delete;

  // Prevents move assignment.
  Config& operator=(Config&&) = delete;

  // Gets the config value of the queried key.
  // NOTE: This function fails if `config_key` doesn't exist in `config_map_`.
  const bool config_value(const std::string_view& config_key) const { return config_map_.at(config_key); }

  static constexpr std::string_view kDoesEnableBackpropagation = "does_enable_backpropagation";
  static constexpr std::string_view kIsTrainingMode = "is_training_mode";

 private:
  Config() {
    config_map_[kDoesEnableBackpropagation] = true;
    config_map_[kIsTrainingMode] = true;
  }

  ~Config() {}

  // Sets the config value of the queried key.
  // NOTE: This function fails if `config_key` doesn't exist in `config_map_`.
  void SetConfigValue(const std::string_view& config_key, const bool config_value) {
    assert((static_cast<void>("`Config::config_map_` must have the value of the key."), config_map_.count(config_key)));
    config_map_[config_key] = config_value;
  }

  std::unordered_map<std::string_view, bool> config_map_;

  friend class UseConfig;
};

class UseConfig {
 public:
  // Preserves the old config value, and changes to a new config value.
  UseConfig(const std::string_view& config_key, const bool new_config_value)
      : config_key_(config_key),
        old_config_value_(Config::instance().config_value(config_key)),
        new_config_value_(new_config_value) {
    Config::instance().SetConfigValue(config_key_, new_config_value_);
  }

  // Restores the old config value.
  ~UseConfig() {
    Config::instance().SetConfigValue(config_key_, old_config_value_);
  }

 private:
  std::string_view config_key_;

  bool old_config_value_;

  bool new_config_value_;
};

}  // namespace tensorward::core
