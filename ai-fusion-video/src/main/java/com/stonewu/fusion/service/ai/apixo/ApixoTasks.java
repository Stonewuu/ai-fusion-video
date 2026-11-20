package com.stonewu.fusion.service.ai.apixo;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;

import java.net.URI;
import java.util.ArrayList;
import java.util.Map;

/** 单次提交保存请求意图，查询只访问固定状态接口；结果下载不携带供应商密钥。 */
public final class ApixoTasks {
    private static final ObjectMapper JSON = new ObjectMapper();

    private ApixoTasks() {
    }

    public static PreparedGenerationCall prepare(ApiConfig api, AiModel model, JSONObject input) {
        var client = new OfficialVideoHttpClient(api);
        var body = new JSONObject().set("request_type", "async").set("input", input);
        var request = client.post(ApixoEndpoints.submit(api, model.getCode()), body, Map.of());
        var intent = JSON.createObjectNode().put("protocol", "apixo").put("model", model.getCode());
        intent.set("body", JSON.valueToTree(body));
        return new PreparedGenerationCall(new GenerationPreparedRequest(null, intent), () -> {
            JSONObject data = data(client.execute(request));
            return GenerationProviderReceipt.accepted(taskId(data));
        });
    }

    public static RemoteGenerationResult query(ApiConfig api, AiModel model, String id) {
        requireTaskId(id);
        JSONObject data = data(new OfficialVideoHttpClient(api).get(ApixoEndpoints.status(api, model.getCode(), id)));
        if (!id.equals(taskId(data))) {
            throw new BusinessException("APIXO 返回的任务编号不一致，请核对原任务");
        }
        return switch (data.getStr("state", "")) {
            case "pending", "processing" -> RemoteGenerationResult.pending(5);
            case "failed" -> RemoteGenerationResult.failed(failureMessage(data));
            case "success" -> completed(data, model.getModelType());
            default -> throw new BusinessException("APIXO 任务状态尚未确认，请稍后查询原任务");
        };
    }

    public static void requireSuccess(JSONObject response) {
        Object code = response.get("code");
        if (!(code instanceof Number number)) {
            throw new BusinessException("APIXO 响应缺少状态码，请稍后重试");
        }
        if (number.intValue() != 200) {
            int status = number.intValue();
            throw GenerationProviderFailure.http(status >= 400 && status <= 599 ? status : 502,
                    response.toString());
        }
    }

    private static JSONObject data(JSONObject response) {
        requireSuccess(response);
        if (!(response.get("data") instanceof JSONObject data)) {
            throw new BusinessException("APIXO 响应缺少任务信息，请稍后查询原任务");
        }
        return data;
    }

    private static String taskId(JSONObject data) {
        Object value = data.get("taskId");
        if (!(value instanceof String id)) {
            throw new BusinessException("APIXO 响应缺少任务编号，请核对任务详情");
        }
        requireTaskId(id);
        return id;
    }

    private static void requireTaskId(String id) {
        if (id == null || id.isBlank() || id.length() > 256) {
            throw new BusinessException("APIXO 任务编号无效，请核对任务详情");
        }
    }

    private static RemoteGenerationResult completed(JSONObject data, int modelType) {
        Object encoded = data.get("resultJson");
        if (!(encoded instanceof String text) || text.length() > 1024 * 1024 || !JSONUtil.isTypeJSONObject(text)) {
            throw new BusinessException("APIXO 任务已完成，但生成结果无法读取，请稍后查询原任务");
        }
        JSONObject result;
        try {
            result = JSONUtil.parseObj(text);
        } catch (RuntimeException invalid) {
            throw new BusinessException("APIXO 生成结果格式无效，请核对任务详情");
        }
        if (!(result.get("resultUrls") instanceof JSONArray urls) || urls.isEmpty() || urls.size() > 64) {
            throw new BusinessException("APIXO 任务已完成，但未返回有效的生成文件");
        }
        var outputs = new ArrayList<RemoteGenerationResult.Output>();
        for (Object value : urls) {
            if (!(value instanceof String url) || !validUrl(url)) {
                throw new BusinessException("APIXO 返回的生成文件地址无效，请核对任务详情");
            }
            outputs.add(modelType == 2 ? RemoteGenerationResult.Output.image(url)
                    : RemoteGenerationResult.Output.video(url, null, null));
        }
        return RemoteGenerationResult.success(outputs);
    }

    private static boolean validUrl(String value) {
        try {
            URI uri = URI.create(value);
            return ("https".equalsIgnoreCase(uri.getScheme()) || "http".equalsIgnoreCase(uri.getScheme()))
                    && uri.getHost() != null && uri.getUserInfo() == null;
        } catch (IllegalArgumentException invalid) {
            return false;
        }
    }

    private static String failureMessage(JSONObject data) {
        return switch (data.getStr("failCode", "")) {
            case "INSUFFICIENT_BALANCE" -> "APIXO 余额不足，请充值后重新生成";
            case "CONTENT_VIOLATION" -> "APIXO 未通过内容审核，请调整提示词或参考素材";
            default -> "APIXO 生成失败，请在平台任务记录中查看原因后重试";
        };
    }
}
