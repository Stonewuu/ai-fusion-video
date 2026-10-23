package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.model.MediaException;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

import java.util.Map;

@RestControllerAdvice(basePackageClasses = CanvasController.class)
@Order(Ordered.HIGHEST_PRECEDENCE)
public class CanvasExceptionHandler {
    public record Failure(CanvasErrorCode errorCode, Map<String, Object> details) {
    }

    public record MediaFailure(String errorCode) {
    }

    @ExceptionHandler(MediaException.class)
    public ResponseEntity<CommonResult<MediaFailure>> media(MediaException failure) {
        CommonResult<MediaFailure> result = CommonResult.error(failure.getCode(), failure.getMessage());
        result.setData(new MediaFailure(failure.getErrorCode()));
        return ResponseEntity.status(failure.getCode()).contentType(MediaType.APPLICATION_JSON).body(result);
    }

    @ExceptionHandler(CanvasException.class)
    public ResponseEntity<CommonResult<Failure>> canvas(CanvasException failure) {
        CommonResult<Failure> result = CommonResult.error(failure.getCode(), failure.getMessage());
        result.setData(new Failure(failure.getErrorCode(), failure.getDetails()));
        return ResponseEntity.status(failure.getCode()).contentType(MediaType.APPLICATION_JSON).body(result);
    }

    @ExceptionHandler({HttpMessageNotReadableException.class, MethodArgumentNotValidException.class,
            MethodArgumentTypeMismatchException.class, MissingServletRequestParameterException.class})
    public ResponseEntity<CommonResult<Failure>> invalidRequest(Exception failure) {
        return canvas(new CanvasException(CanvasErrorCode.INVALID_COMMAND, "请求格式或参数无效"));
    }
}
