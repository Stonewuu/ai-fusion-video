package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.service.canvas.model.CanvasActor;
import jakarta.validation.constraints.NotNull;

import java.util.Objects;

public record GenerationClientScope(@NotNull Long userId, Long teamId) {
    public void require(CanvasActor actor) {
        if (!Objects.equals(userId, actor.userId()) || !Objects.equals(teamId, actor.teamId())) {
            throw new GenerationJobException(409, "GENERATION_SCOPE_CHANGED",
                    "当前账号或团队已变化，请回到原提交范围确认任务");
        }
    }
}
