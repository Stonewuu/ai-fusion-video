package com.stonewu.fusion.service.generation.job;

/** 同一资产执行任务首次受理时保存的输入。 */
public record AssetImageReferenceSnapshot(String submissionId, String inputJson) {
}
