package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.NormalizedModelUsage;
import reactor.core.publisher.Mono;

/** Replaceable downstream boundary for idempotent model-usage settlement. */
public interface ModelUsageSettlementPort {

    Mono<String> settle(String idempotencyKey, NormalizedModelUsage usage);
}
