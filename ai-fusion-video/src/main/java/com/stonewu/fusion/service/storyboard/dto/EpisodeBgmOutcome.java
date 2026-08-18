package com.stonewu.fusion.service.storyboard.dto;

import java.nio.file.Path;

/**
 * 合成后配乐步骤的结果。
 *
 * @param scoredVideoFile 已混入配乐的成片文件；为 null 表示本次跳过配乐（如超时长上限）
 * @param musicAudioUrl   配乐音轨持久化后的URL
 * @param musicLicenseId  配乐音轨授权凭证ID（商用留档）
 * @param sfxAudioUrl     音效音轨持久化后的URL（未开启或跳过时为 null）
 * @param sfxLicenseId    音效音轨授权凭证ID（商用留档）
 * @param skippedReason   跳过配乐的原因；为 null 表示配乐已生成
 */
public record EpisodeBgmOutcome(
        Path scoredVideoFile,
        String musicAudioUrl,
        String musicLicenseId,
        String sfxAudioUrl,
        String sfxLicenseId,
        String skippedReason) {

    public static EpisodeBgmOutcome skipped(String reason) {
        return new EpisodeBgmOutcome(null, null, null, null, null, reason);
    }
}
