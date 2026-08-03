package com.stonewu.fusion.service.generation.image.strategy.openai;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolAdapter;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolContext;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolSupport;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageRequest;

import org.springframework.stereotype.Component;

import java.util.List;

/** Standard OpenAI Images API generations/edits adapter. */
@Component
public class OpenAiImageProtocolAdapter implements OpenAiCompatibleImageProtocolAdapter {

    private final OpenAiCompatibleImageProtocolSupport support;

    public OpenAiImageProtocolAdapter(OpenAiCompatibleImageProtocolSupport support) {
        this.support = support;
    }

    @Override
    public String getProtocol() {
        return "openai";
    }

    @Override
    public OpenAiCompatibleImageRequest buildRequest(OpenAiCompatibleImageProtocolContext context) {
        if (context.asyncMode()) {
            throw new BusinessException("OpenAI 官方 Images API 不支持异步任务模式");
        }
        return context.hasReferenceImages()
                ? support.buildOpenAiEditsRequest(context)
                : support.buildOpenAiGenerationsRequest(context);
    }

    @Override
    public List<String> parseImageUrls(OpenAiCompatibleImageProtocolContext context, String responseBody) {
        return support.parseOpenAiImageUrls(responseBody);
    }

    @Override
    public boolean isSingleImagePerRequest(OpenAiCompatibleImageProtocolContext context) {
        Boolean configured = support.getBoolean(context.modelConfig(),
                "singleImagePerRequest", "disableBatchCount");
        return Boolean.TRUE.equals(configured);
    }
}
