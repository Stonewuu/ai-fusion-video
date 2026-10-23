package com.stonewu.fusion.controller.script;

import com.stonewu.fusion.controller.canvas.CanvasExceptionHandler;
import org.springframework.core.annotation.Order;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Order(0)
@RestControllerAdvice(assignableTypes = ScriptController.class)
public class ScriptCommandExceptionHandler extends CanvasExceptionHandler {
}
