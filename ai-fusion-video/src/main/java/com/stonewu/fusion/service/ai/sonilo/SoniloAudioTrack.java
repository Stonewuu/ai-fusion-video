package com.stonewu.fusion.service.ai.sonilo;

/**
 * Sonilo 生成结果中的一条音轨。
 *
 * @param url       音频下载地址（预签名 URL，自带鉴权）
 * @param licenseId 该音轨的授权凭证ID，用于商用留档
 */
public record SoniloAudioTrack(String url, String licenseId) {
}
