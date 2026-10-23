package com.stonewu.fusion.controller.generation;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.canvas.CanvasExceptionHandler;
import com.stonewu.fusion.service.generation.job.GenerationJobException;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Order(Ordered.HIGHEST_PRECEDENCE)
@RestControllerAdvice(basePackageClasses = GenerationExecutionController.class)
public class GenerationExceptionHandler extends CanvasExceptionHandler {
    @ExceptionHandler(GenerationJobException.class)
    public ResponseEntity<CommonResult<MediaFailure>> generation(GenerationJobException failure) {
        CommonResult<MediaFailure> result = CommonResult.error(failure.getCode(), failure.getMessage());
        result.setData(new MediaFailure(failure.getErrorCode()));
        return ResponseEntity.status(failure.getCode()).body(result);
    }
}
