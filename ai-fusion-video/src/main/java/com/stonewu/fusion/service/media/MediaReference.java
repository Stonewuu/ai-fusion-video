package com.stonewu.fusion.service.media;

import com.stonewu.fusion.service.media.model.MediaException;

/**
 * 持久字段保存稳定资源标识；访问票据只在传输时签发，不能写回业务正文。
 */
public final class MediaReference {
    private static final String PREFIX = "media://";

    private MediaReference() {
    }

    public static boolean managed(String value) {
        return value != null && value.regionMatches(true, 0, PREFIX, 0, PREFIX.length());
    }

    public static String id(String reference) {
        if (!managed(reference)) {
            throw MediaException.invalid("不是托管媒体引用");
        }
        String id = reference.substring(PREFIX.length());
        if (!reference.startsWith(PREFIX) || !id.matches(
                "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")) {
            throw MediaException.invalid("托管媒体引用格式无效");
        }
        return id;
    }

    public static String of(String id) {
        String result = PREFIX + id;
        id(result);
        return result;
    }
}
