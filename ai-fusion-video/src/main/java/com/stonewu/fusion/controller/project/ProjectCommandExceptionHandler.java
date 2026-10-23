package com.stonewu.fusion.controller.project;

import com.stonewu.fusion.controller.canvas.CanvasExceptionHandler;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice(basePackageClasses = ProjectController.class)
@Order(Ordered.HIGHEST_PRECEDENCE)
public class ProjectCommandExceptionHandler extends CanvasExceptionHandler {
}
