#ifndef UCM_POSIX_PACKED_LAYOUT_H
#define UCM_POSIX_PACKED_LAYOUT_H

#include <algorithm>
#include <climits>
#include <cstdint>
#include "global_config.h"
#include "posix_file.h"
#include "type/dictionary.h"

namespace UC::PosixStore {
inline Status ParsePackedLayout(const Detail::Dictionary& input, Config& config)
{
    if (!input.Contains("posix_packed_layer_bytes")) { return Status::OK(); }
    std::vector<ssize_t> lengths;
    try { input.Get("posix_packed_layer_bytes", lengths); }
    catch (const std::bad_any_cast&) {
        return Status::InvalidParam("packed layer bytes must be a nonempty integer list");
    }
    if (lengths.empty() || lengths.size() > IOV_MAX) {
        return Status::InvalidParam("invalid packed layer count");
    }
    for (const auto length : lengths) {
        if (length <= 0 || length % 4096 != 0 ||
            static_cast<uint64_t>(length) > UINT32_MAX - config.diskBlockBytes) {
            return Status::InvalidParam("packed layer lengths must be 4 KiB aligned and fit one AIO read");
        }
        config.layerOffsets.push_back(config.diskBlockBytes);
        config.layerBytes.push_back(static_cast<size_t>(length));
        config.diskBlockBytes += static_cast<size_t>(length);
    }
    if (config.deviceId >= 0) {
        if (!config.shardSize || config.tensorSize != config.shardSize ||
            config.blockSize % config.shardSize ||
            config.blockSize / config.shardSize != lengths.size()) {
            return Status::InvalidParam("packed layout requires one contiguous host slice per layer");
        }
        for (const auto length : config.layerBytes) {
            if (length > config.shardSize) {
                return Status::InvalidParam("packed layer exceeds its L1 slice");
            }
        }
    }
    return Status::OK();
}

inline Status PreparePackedNamespace(Config& config)
{
    if (config.layerBytes.empty()) { return Status::OK(); }
    // Exact lengths, including their count, isolate incompatible offsets without
    // depending on a process-specific hash or a writable format manifest.
    std::vector<std::string> parts{
        ".ucm-packed-v1-" + std::to_string(config.layerBytes.size())};
    for (size_t start = 0; start < config.layerBytes.size(); start += 16) {
        std::string part = "layers";
        for (size_t i = start; i < std::min(start + 16, config.layerBytes.size()); ++i) {
            part += "-" + std::to_string(config.layerBytes[i]);
        }
        parts.push_back(std::move(part));
    }
    for (auto& backend : config.storageBackends) {
        for (const auto& part : parts) {
            backend += "/" + part;
            auto status = PosixFile{backend}.MkDir();
            if (status.Failure() && status != Status::DuplicateKey()) { return status; }
        }
    }
    return Status::OK();
}
}  // namespace UC::PosixStore
#endif
