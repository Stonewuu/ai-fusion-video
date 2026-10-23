package com.stonewu.fusion.controller.storyboard;

import com.stonewu.fusion.controller.canvas.CanvasExceptionHandler;
import org.springframework.core.annotation.Order;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Order(-10)
@RestControllerAdvice(assignableTypes = StoryboardController.class)
public class StoryboardCommandExceptionHandler extends CanvasExceptionHandler {
}
