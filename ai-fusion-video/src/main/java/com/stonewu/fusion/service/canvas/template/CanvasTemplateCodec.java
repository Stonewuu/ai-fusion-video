package com.stonewu.fusion.service.canvas.template;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;

import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.MAX_BYTES;
import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.Template;

@Component
@RequiredArgsConstructor
public class CanvasTemplateCodec {
    private final ObjectMapper mapper;
    private final CanvasTemplateValidation validation;

    public Template parse(String content) {
        if (content == null || content.getBytes(StandardCharsets.UTF_8).length > MAX_BYTES) {
            throw new BusinessException(400, "模板文件为空或超过 16 MiB");
        }
        Template template;
        try {
            template = mapper.readerFor(Template.class).with(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES,
                    DeserializationFeature.FAIL_ON_TRAILING_TOKENS).readValue(content);
        } catch (JsonProcessingException failure) {
            throw new BusinessException(400, "模板 JSON 格式或字段不符合协议");
        }
        validation.validate(template);
        return template;
    }
}
