package com.stonewu.fusion.controller.media;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.model.MediaException;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;

import java.io.IOException;

@RestControllerAdvice(basePackageClasses = MediaResourceController.class)
@Order(Ordered.HIGHEST_PRECEDENCE)
public class MediaExceptionHandler {
    public record Failure(String errorCode) {
    }

    @ExceptionHandler(MediaException.class)
    public ResponseEntity<CommonResult<Failure>> media(MediaException error) {
        CommonResult<Failure> body = CommonResult.error(error.getCode(), error.getMessage());
        body.setData(new Failure(error.getErrorCode()));
        return ResponseEntity.status(error.getCode()).header("Cache-Control", "no-store").body(body);
    }

    @ExceptionHandler(CanvasException.class)
    public ResponseEntity<CommonResult<Failure>> scope(CanvasException error) {
        return media(new MediaException(error.getCode(), error.getErrorCode().name(), error.getMessage()));
    }

    @ExceptionHandler(IOException.class)
    public ResponseEntity<CommonResult<Failure>> io(IOException error) {
        return media(new MediaException(502, "MEDIA_IO_FAILED", "媒体读写未完成，请重试"));
    }

    @ExceptionHandler({MissingServletRequestParameterException.class, MethodArgumentTypeMismatchException.class,
            MissingServletRequestPartException.class, MethodArgumentNotValidException.class, HttpMessageNotReadableException.class})
    public ResponseEntity<CommonResult<Failure>> invalid(Exception error) {
        return media(MediaException.invalid("媒体请求参数无效"));
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<CommonResult<Failure>> tooLarge(MaxUploadSizeExceededException error) {
        return media(new MediaException(413, "MEDIA_TOO_LARGE", "媒体文件超过大小限制"));
    }
}
