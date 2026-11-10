package com.stonewu.fusion.service.generation.audio;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.LocalDateTime;
import java.util.List;

public final class AudioTaskData {
    private AudioTaskData() {
    }

    public record Input(@NotBlank @Size(max = 16000) String text,
                        @NotNull @Size(max = 6) List<@NotBlank String> media,
                        @Size(max = 100) String voice) {
    }

    public record Create(@NotNull Long modelId, @NotBlank @Size(max = 80) String requestId,
                         @NotNull @Valid Input input) {
    }

    public record Route(String platform, String apiUrl, String projectId) {
    }

    public record Result(String text, List<String> audio) {
        public Result {
            audio = List.copyOf(audio);
        }
    }

    public record View(String id, Long modelId, String modelName, String state,
                       Result result, String errorMessage, boolean canRetrySave,
                       LocalDateTime createTime, LocalDateTime updateTime) {
    }

    /** 先保留完整响应，再解析音频；解析和保存重试不会再次调用模型。 */
    public record Received(String operation, String body) {
    }
}
