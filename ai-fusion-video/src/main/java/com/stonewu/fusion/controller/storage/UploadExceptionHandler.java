package com.stonewu.fusion.controller.storage;

import com.stonewu.fusion.controller.canvas.CanvasExceptionHandler;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice(assignableTypes = FileUploadController.class)
@Order(Ordered.HIGHEST_PRECEDENCE)
public class UploadExceptionHandler extends CanvasExceptionHandler {
}
