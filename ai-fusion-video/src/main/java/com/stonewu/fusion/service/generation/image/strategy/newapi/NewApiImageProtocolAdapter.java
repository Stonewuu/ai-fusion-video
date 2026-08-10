package com.stonewu.fusion.service.generation.image.strategy.newapi;

import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolAdapter;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolContext;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolSupport;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageRequest;
import org.springframework.stereotype.Component;

import java.util.List;

/** OpenAI-shaped, non-official image APIs such as NewAPI async task endpoints. */
@Component
public class NewApiImageProtocolAdapter implements OpenAiCompatibleImageProtocolAdapter {

    private final OpenAiCompatibleImageProtocolSupport support;

    public NewApiImageProtocolAdapter(OpenAiCompatibleImageProtocolSupport support) {
        this.support = support;
    }

    @Override
    public String getProtocol() {
        return "newapi";
    }

    @Override
    public OpenAiCompatibleImageRequest buildRequest(OpenAiCompatibleImageProtocolContext context) {
        Boolean referencesViaGenerations = support.getBoolean(context.modelConfig(),
                "referenceImagesViaGenerations", "useGenerationsForReferenceImages");
        return context.asyncMode() || !context.hasReferenceImages() || Boolean.TRUE.equals(referencesViaGenerations)
                ? support.buildNewApiGenerationsRequest(context)
                : support.buildNewApiEditsRequest(context);
    }

    @Override
    public List<String> parseImageUrls(OpenAiCompatibleImageProtocolContext context, String responseBody) {
        return support.parseImageUrls(responseBody);
    }

    @Override
    public boolean isSingleImagePerRequest(OpenAiCompatibleImageProtocolContext context) {
        return Boolean.TRUE.equals(support.getBoolean(context.modelConfig(),
                "singleImagePerRequest", "disableBatchCount"));
    }
}
