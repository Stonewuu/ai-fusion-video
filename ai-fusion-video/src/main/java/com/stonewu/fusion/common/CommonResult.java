package com.stonewu.fusion.common;

import lombok.Data;

import java.io.Serializable;

/**
 * 通用 API 响应封装
 */
@Data
public class CommonResult<T> implements Serializable {

    private Integer code;
    private String msg;
    private T data;

    public static <T> CommonResult<T> success(T data) {
        CommonResult<T> result = new CommonResult<>();
        result.setCode(0);
        result.setMsg("success");
        result.setData(data);
        return result;
    }

    public static <T> CommonResult<T> success() {
        return success(null);
    }

    public static <T> CommonResult<T> error(Integer code, String msg) {
        CommonResult<T> result = new CommonResult<>();
        result.setCode(code);
        result.setMsg(msg);
        return result;
    }

    public static <T> CommonResult<T> error(String msg) {
        return error(500, msg);
    }
}
