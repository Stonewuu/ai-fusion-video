package com.stonewu.fusion.service.generation.image.strategy.agnes;

import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolAdapter;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolContext;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolSupport;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageRequest;

import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Agnes Image 2.0/2.1 adapter.
 *
 * <p>Both model versions intentionally share this adapter.  Agnes exposes
 * image-to-image through the generations endpoint and places reference images
 * and response format under {@code extra_body}; Data URIs are passed through
 * unchanged by the support layer.</p>
 */
@Component
public class AgnesImageProtocolAdapter implements OpenAiCompatibleImageProtocolAdapter {

    private final OpenAiCompatibleImageProtocolSupport support;

    public AgnesImageProtocolAdapter(OpenAiCompatibleImageProtocolSupport support) {
        this.support = support;
    }

    @Override
    public String getProtocol() {
        return "agnes";
    }

    @Override
    public OpenAiCompatibleImageRequest buildRequest(OpenAiCompatibleImageProtocolContext context) {
        return support.buildAgnesGenerationsRequest(context);
    }

    @Override
    public List<String> parseImageUrls(OpenAiCompatibleImageProtocolContext context, String responseBody) {
        return support.parseAgnesImageUrls(responseBody);
    }

    @Override
    public boolean isSingleImagePerRequest(OpenAiCompatibleImageProtocolContext context) {
        Boolean configured = support.getBoolean(context.modelConfig(),
                "singleImagePerRequest", "disableBatchCount");
        return configured != null ? configured : true;
    }
}
