package com.stonewu.fusion.service.ai.catalog;

import java.util.List;

public record ProviderDiscoveryResult(String providerId, boolean keyVerified, String notice,
                                      List<ProviderModelCandidate> models) {
}
