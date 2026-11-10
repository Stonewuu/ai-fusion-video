package com.stonewu.fusion.service.ai.google;

import java.io.IOException;
import java.io.UncheckedIOException;

/** 请求已进入网络传输但未收到完整回执，不能自动重发。 */
public final class GoogleRequestUncertainException extends UncheckedIOException {
    public GoogleRequestUncertainException(IOException cause) {
        super("Google 调用结果待确认", cause);
    }
}
