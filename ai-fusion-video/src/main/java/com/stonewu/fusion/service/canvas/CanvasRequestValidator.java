package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import jakarta.validation.Validator;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.regex.Pattern;

@Component
@RequiredArgsConstructor
public class CanvasRequestValidator {
    private static final Pattern ID = Pattern.compile("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}");
    private final Validator validator;

    public void id(String value) {
        if (value == null || !ID.matcher(value).matches()) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "画布 ID 必须是小写 UUID");
        }
    }

    public void validate(Object value) {
        if (value == null) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "请求不能为空");
        }
        var failures = validator.validate(value);
        if (!failures.isEmpty()) {
            throw new CanvasException(400, CanvasErrorCode.INVALID_COMMAND, "画布请求参数无效",
                    Map.of("fields",
                            failures.stream().map(item -> item.getPropertyPath().toString()).sorted().toList()));
        }
    }
}
