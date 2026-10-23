package com.stonewu.fusion.service.media;

import com.stonewu.fusion.mapper.media.MediaPresetBindingMapper;

import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;

/**
 * 只有专用入口登记的内置图片才授予全局读取；普通上传和猜测系统配置键均不能扩大权限。
 */
@Component
@RequiredArgsConstructor
public class MediaPresetReferences {
    private final MediaPresetBindingMapper bindings;

    public boolean canRead(String mediaId) {
        return bindings.readable(mediaId, MediaReference.of(mediaId)) != null;
    }
}
