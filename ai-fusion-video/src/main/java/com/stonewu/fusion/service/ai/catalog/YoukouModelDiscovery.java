package com.stonewu.fusion.service.ai.catalog;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.youkou.YoukouContract;
import com.stonewu.fusion.service.ai.youkou.YoukouHttpClient;
import com.stonewu.fusion.service.ai.youkou.YoukouModelCapabilities;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashSet;

/** 按当前密钥的模型身份关联目录与报价，名称只用于展示和按合同调用。 */
@Component
public class YoukouModelDiscovery implements ProviderModelDiscovery {
    @Override
    public String type() {
        return "youkou";
    }

    @Override
    public ProviderDiscoveryResult discover(ApiConfig config, ProviderDefinition provider) {
        var client = new YoukouHttpClient(config);
        var models = client.get("/models");
        var catalog = client.get("/catalog");
        var prices = client.get("/pricing");
        if (!"platform-async-20261019".equals(catalog.path("contract_version").asText())) {
            throw new BusinessException("贰春接口合同已变化，请更新接入后再添加模型");
        }
        if (!catalog.path("catalog_version").equals(prices.path("catalog_version"))) {
            throw new BusinessException("平台模型目录已更新，请重新读取模型");
        }
        var authorized = new HashSet<String>();
        for (JsonNode entry : rows(models)) {
            authorized.add(required(entry, "id"));
        }
        var priced = new HashSet<String>();
        for (JsonNode entry : rows(prices)) {
            priced.add(required(entry, "public_model_id"));
        }
        var result = new ArrayList<ProviderModelCandidate>();
        var seen = new HashSet<String>();
        for (JsonNode entry : rows(catalog)) {
            String id = required(entry, "id");
            String identity = required(entry, "public_model_id");
            if (!authorized.contains(id)) {
                continue;
            }
            if (!seen.add(id) || !identity.matches("mdl_[a-f0-9]{32}")) {
                throw new BusinessException("平台模型目录包含无效或重复的模型身份，请稍后重试");
            }
            int type = switch (entry.path("output_type").asText()) {
                case "text" -> 1;
                case "image" -> 2;
                case "video" -> 3;
                default -> 0;
            };
            if (type == 0) {
                continue;
            }
            String endpoint = switch (type) {
                case 1 -> "/v1/chat/completions";
                case 2 -> "/v1/images/generations";
                default -> "/v1/videos";
            };
            if (!YoukouContract.values(entry.path("endpoints")).contains(endpoint)) {
                continue;
            }
            var options = YoukouModelCapabilities.configuration(entry, endpoint);
            String hint = YoukouModelCapabilities.unavailable(entry, options, endpoint);
            if (!priced.contains(identity)) {
                hint = "平台尚未返回此模型的报价信息，请稍后重新读取";
            }
            result.add(new ProviderModelCandidate(id, required(entry, "display_name"), type,
                    type == 1 ? "openai_compatible" : "youkou", null, options.toString(), hint,
                    type == 1 ? YoukouModelCapabilities.text(entry) : null));
        }
        return new ProviderDiscoveryResult(provider.id(), true, "选择需要的对话、图片或视频模型。", result);
    }

    private ArrayNode rows(JsonNode response) {
        if (!(response.path("data") instanceof ArrayNode rows)) {
            throw new BusinessException("贰春模型目录格式无效，请稍后重试");
        }
        return rows;
    }

    private String required(JsonNode node, String field) {
        if (!node.path(field).isTextual() || node.get(field).asText().isBlank()) {
            throw new BusinessException("贰春模型目录缺少必要信息，请稍后重新读取");
        }
        return node.get(field).asText();
    }
}
