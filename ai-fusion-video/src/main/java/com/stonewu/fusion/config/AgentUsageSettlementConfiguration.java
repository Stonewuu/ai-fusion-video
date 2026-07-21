package com.stonewu.fusion.config;

import com.stonewu.fusion.service.ai.run.AuditLedgerModelUsageSettlementAdapter;
import com.stonewu.fusion.service.ai.run.ModelUsageSettlementPort;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration(proxyBeanMethods = false)
public class AgentUsageSettlementConfiguration {

    @Bean
    @ConditionalOnMissingBean(ModelUsageSettlementPort.class)
    ModelUsageSettlementPort modelUsageSettlementPort() {
        return new AuditLedgerModelUsageSettlementAdapter();
    }
}
