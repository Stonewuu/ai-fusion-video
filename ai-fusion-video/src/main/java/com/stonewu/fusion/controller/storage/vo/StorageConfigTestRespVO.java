package com.stonewu.fusion.controller.storage.vo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 存储配置连接测试响应。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class StorageConfigTestRespVO {

    private Boolean success;

    private String message;

    private String publicUrl;
}
