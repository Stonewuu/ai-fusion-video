package com.stonewu.fusion.service.generation.video.strategy.newapi;

import com.stonewu.fusion.common.BusinessException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * NewAPI 视频协议路由器。
 */
@Component
@RequiredArgsConstructor
public class NewApiVideoProtocolRouter {

    private final List<NewApiVideoProtocolAdapter> adapters;

    private volatile Map<String, NewApiVideoProtocolAdapter> adapterMap;

    public NewApiVideoProtocolAdapter resolve(NewApiVideoProtocolContext context) {
        Map<String, NewApiVideoProtocolAdapter> map = getAdapterMap();
        String protocol = context.metadata() != null ? context.metadata().modelProtocol() : null;
        NewApiVideoProtocolAdapter adapter = map.get(protocol);
        if (adapter != null) {
            return adapter;
        }
        throw new BusinessException("New API 未配置对应的视频协议适配器: " + protocol);
    }

    public Set<String> getSupportedProtocols() {
        return Set.copyOf(getAdapterMap().keySet());
    }

    private Map<String, NewApiVideoProtocolAdapter> getAdapterMap() {
        if (adapterMap == null) {
            synchronized (this) {
                if (adapterMap == null) {
                    Map<String, NewApiVideoProtocolAdapter> resolved = new LinkedHashMap<>();
                    for (NewApiVideoProtocolAdapter adapter : adapters) {
                        resolved.putIfAbsent(adapter.getProtocol(), adapter);
                    }
                    adapterMap = resolved;
                }
            }
        }
        return adapterMap;
    }
}
