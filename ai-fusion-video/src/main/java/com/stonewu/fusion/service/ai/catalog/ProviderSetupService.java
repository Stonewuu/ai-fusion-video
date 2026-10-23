package com.stonewu.fusion.service.ai.catalog;

import cn.hutool.crypto.digest.DigestUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.ProviderSetupReqVO;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.mapper.ai.ApiConfigMapper;
import com.stonewu.fusion.security.SecurityUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashSet;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ProviderSetupService {
    private final ProviderDiscoveryService discovery;
    private final ProviderSetupWriter writer;
    private final ApiConfigMapper configs;
    private final ObjectMapper json;

    @Transactional(propagation = Propagation.NEVER)
    public ProviderSetupResult apply(ProviderSetupReqVO request) {
        if (new HashSet<>(request.modelIds()).size() != request.modelIds().size()) {
            throw new BusinessException(400, "模型选择不能重复");
        }
        String requestKey = DigestUtil.sha256Hex(SecurityUtils.requireCurrentUserId() + ":" + request.requestId());
        String requestHash = requestHash(request);
        ProviderSetupResult replay = replay(requestKey, requestHash, request.modelIds().size());
        if (replay != null) {
            return replay;
        }
        var config = discovery.resolveConfiguration(request.configuration());
        var candidates = discovery.preview(request.configuration()).models().stream()
                .collect(Collectors.toMap(ProviderModelCandidate::id, Function.identity()));
        List<ProviderModelCandidate> selected = request.modelIds().stream().map(id -> {
            var candidate = candidates.get(id);
            if (candidate == null) {
                throw new BusinessException(409, "所选模型当前不可用，请重新读取模型列表");
            }
            if (candidate.configurationHint() != null) {
                throw new BusinessException(400, candidate.name() + "：" + candidate.configurationHint());
            }
            return candidate;
        }).toList();
        try {
            return writer.save(config, selected, requestKey, requestHash);
        } catch (DuplicateKeyException concurrentRequest) {
            ProviderSetupResult completed = replay(requestKey, requestHash, selected.size());
            if (completed == null) {
                throw concurrentRequest;
            }
            return completed;
        }
    }

    private ProviderSetupResult replay(String key, String hash, int selectedCount) {
        var existing = configs.selectOne(new LambdaQueryWrapper<ApiConfig>().eq(ApiConfig::getProviderSetupKey, key));
        if (existing == null) {
            return null;
        }
        if (!hash.equals(existing.getProviderSetupHash())) {
            throw new BusinessException(409, "此配置请求已经完成，修改选择后请开始新的配置请求");
        }
        return new ProviderSetupResult(existing.getId(), 0, selectedCount);
    }

    private String requestHash(ProviderSetupReqVO request) {
        try {
            return DigestUtil.sha256Hex(json.writeValueAsString(List.of(request.configuration(),
                    request.modelIds().stream().sorted().toList())));
        } catch (JsonProcessingException invalid) {
            throw new IllegalArgumentException("快速配置请求无法编码", invalid);
        }
    }
}
