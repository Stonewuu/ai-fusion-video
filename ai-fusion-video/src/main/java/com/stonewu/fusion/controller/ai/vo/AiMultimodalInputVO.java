package com.stonewu.fusion.controller.ai.vo;

import lombok.Data;

/** One binary input attached to an assistant user message. */
@Data
public class AiMultimodalInputVO {

    private String id;
    private String name;
    private String inputType;
    private String mimeType;
    private String transport;
    private String url;
    private String data;
    /** 应用侧持久化媒体地址，仅用于消息回显，不参与模型请求。 */
    private String resourceUrl;
    private Long size;
}
