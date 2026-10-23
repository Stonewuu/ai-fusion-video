package com.stonewu.fusion.service.media.job;

import com.stonewu.fusion.service.media.model.MediaSourceReference;

import java.util.List;

public record ComposeSnapshot(List<MediaSourceReference> clips, Bgm bgm) {
    public record Bgm(long apiConfigId, String configHash, String prompt, double volume, boolean sfx,
                      int timeoutSeconds, int pollSeconds) {
    }
}
