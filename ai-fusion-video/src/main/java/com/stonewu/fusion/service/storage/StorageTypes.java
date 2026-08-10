package com.stonewu.fusion.service.storage;

import cn.hutool.core.util.StrUtil;

/**
 * Storage type and legacy provider helpers.
 */
public final class StorageTypes {

    public static final String LOCAL = "local";
    public static final String S3 = "s3";

    private StorageTypes() {
    }

    public static String normalizeType(String type) {
        if (StrUtil.isBlank(type)) {
            return LOCAL;
        }
        return switch (type.trim()) {
            case "aliyun_oss", "tencent_cos" -> S3;
            default -> type.trim();
        };
    }

    public static String legacyProvider(String type) {
        if (StrUtil.isBlank(type)) {
            return null;
        }
        return switch (type.trim()) {
            case "aliyun_oss" -> "aliyun_oss";
            case "tencent_cos" -> "tencent_cos";
            default -> null;
        };
    }

    public static boolean isS3Like(String type) {
        return S3.equals(normalizeType(type));
    }
}
