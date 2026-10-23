package com.stonewu.fusion.service.ai.catalog;

import com.stonewu.fusion.entity.ai.ApiConfig;

public interface ProviderModelDiscovery {
    String type();

    ProviderDiscoveryResult discover(ApiConfig config, ProviderDefinition provider);
}
