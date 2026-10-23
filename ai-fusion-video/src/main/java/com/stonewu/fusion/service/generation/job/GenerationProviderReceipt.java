package com.stonewu.fusion.service.generation.job;

/**
 * 异步回执提供供应商标识；同步回执提供实际响应，二者都必须先落库。
 */
public record GenerationProviderReceipt(String providerTaskId, RemoteGenerationResult result) {
    public GenerationProviderReceipt {
        if ((providerTaskId == null || providerTaskId.isBlank()) && result == null) {
            throw GenerationJobException.invalid("供应商未返回可追踪回执或同步结果");
        }
        if (providerTaskId != null && providerTaskId.length() > 512) {
            throw GenerationJobException.invalid("供应商标识过长");
        }
        if (result != null && result.state() == RemoteGenerationResult.State.PENDING) {
            throw GenerationJobException.invalid("同步回执不能包含尚未受理的等待状态");
        }
    }

    public static GenerationProviderReceipt accepted(String id) {
        return new GenerationProviderReceipt(id, null);
    }

    public static GenerationProviderReceipt completed(RemoteGenerationResult result) {
        return new GenerationProviderReceipt(null, result);
    }
}
