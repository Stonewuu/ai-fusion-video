package com.stonewu.fusion.service.generation.job;

import java.util.List;

/**
 * 供应商的一次查询结果；只有托管完成后才能转换成用户可用的输出。
 */
public record RemoteGenerationResult(State state, List<Output> outputs, String error, int nextPollSeconds) {
    public enum State {PENDING, SUCCEEDED, FAILED, CANCELLED}

    public RemoteGenerationResult {
        if (state == null) {
            throw new IllegalArgumentException("缺少供应商结果状态");
        }
        outputs = outputs == null ? List.of() : List.copyOf(outputs);
        if (state == State.SUCCEEDED && outputs.isEmpty()) {
            throw GenerationJobException.invalid("供应商成功响应没有可用输出");
        }
        if (state != State.SUCCEEDED && !outputs.isEmpty()) {
            throw GenerationJobException.invalid("非成功结果不能发布输出");
        }
        if (outputs.size() > 64) {
            throw GenerationJobException.invalid("供应商输出数量超过接收上限");
        }
        nextPollSeconds = Math.max(1, Math.min(60, nextPollSeconds));
    }

    public static RemoteGenerationResult pending(int seconds) {
        return new RemoteGenerationResult(State.PENDING, List.of(), null, seconds);
    }

    public static RemoteGenerationResult success(List<Output> outputs) {
        return new RemoteGenerationResult(State.SUCCEEDED, outputs, null, 1);
    }

    public static RemoteGenerationResult failed(String error) {
        return new RemoteGenerationResult(State.FAILED, List.of(), error, 1);
    }

    public static RemoteGenerationResult cancelled(String error) {
        return new RemoteGenerationResult(State.CANCELLED, List.of(), error, 1);
    }

    public record Output(String media, String cover, String firstFrame, String lastFrame, Integer durationSeconds) {
        public Output {
            if (media == null || media.isBlank()) {
                throw GenerationJobException.invalid("供应商输出缺少媒体来源");
            }
        }

        public static Output image(String media) {
            return new Output(media, null, null, null, null);
        }

        public static Output video(String media, String cover, Integer duration) {
            return new Output(media, cover, null, null, duration);
        }
    }
}
