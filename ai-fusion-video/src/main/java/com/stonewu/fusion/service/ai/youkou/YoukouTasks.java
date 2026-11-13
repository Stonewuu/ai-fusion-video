package com.stonewu.fusion.service.ai.youkou;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;

import java.net.URI;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

public final class YoukouTasks {
    private YoukouTasks() {
    }

    public static PreparedGenerationCall prepare(YoukouHttpClient client, YoukouContract contract,
                                                  String endpoint, ObjectNode body) {
        String key = UUID.randomUUID().toString();
        ObjectNode intent = YoukouContract.JSON.createObjectNode().put("protocol", "youkou")
                .put("idempotencyKey", key).put("path", endpoint);
        intent.set("body", body.deepCopy());
        return new PreparedGenerationCall(new GenerationPreparedRequest(null, intent), () -> {
            ObjectNode receipt = client.create(endpoint.substring(3), body, key);
            String id = receipt.path("id").asText();
            validateTask(receipt, id, contract);
            return GenerationProviderReceipt.accepted(id);
        });
    }

    public static RemoteGenerationResult query(YoukouHttpClient client, String id, YoukouContract contract) {
        requireId(id);
        var reply = client.read("/tasks/" + id);
        if (reply.status() == 429) {
            return RemoteGenerationResult.pending(reply.retrySeconds());
        }
        ObjectNode task = reply.body();
        validateTask(task, id, contract);
        return switch (task.path("status").asText()) {
            case "queued", "in_progress" -> RemoteGenerationResult.pending(reply.retrySeconds());
            case "failed" -> RemoteGenerationResult.failed(
                    GenerationProviderFailure.http(502, task.toString()).reason());
            case "completed" -> completed(client, id, contract, task);
            default -> throw new BusinessException("贰春任务状态尚未确认，请稍后查看原任务");
        };
    }

    private static RemoteGenerationResult completed(YoukouHttpClient client, String id,
                                                     YoukouContract contract, ObjectNode task) {
        if (contract.kind().equals("image")) {
            requirePath(task, "result_url", "/v1/tasks/" + id + "/result");
            JsonNode data = client.get("/tasks/" + id + "/result").path("data");
            if (!data.isArray() || data.isEmpty()) {
                throw new BusinessException("贰春图片任务已完成，但图片结果尚不可读取");
            }
            List<RemoteGenerationResult.Output> outputs = new ArrayList<>();
            for (JsonNode item : data) {
                String source;
                if (item.path("url").isTextual()) {
                    source = item.get("url").asText();
                    URI uri;
                    try {
                        uri = URI.create(source);
                    } catch (IllegalArgumentException invalid) {
                        throw new BusinessException("贰春返回的图片地址无效");
                    }
                    if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null
                            || uri.getUserInfo() != null) {
                        throw new BusinessException("贰春返回的图片地址无效");
                    }
                } else if (item.path("b64_json").isTextual()) {
                    source = imageData(item.get("b64_json").asText());
                } else {
                    throw new BusinessException("贰春图片结果缺少文件内容");
                }
                outputs.add(RemoteGenerationResult.Output.image(source));
            }
            return RemoteGenerationResult.success(outputs);
        }
        requirePath(task, "content_url", "/v1/tasks/" + id + "/content");
        Integer duration = task.path("seconds").canConvertToInt() && task.path("seconds").asInt() > 0
                ? task.path("seconds").asInt() : null;
        return RemoteGenerationResult.success(List.of(
                RemoteGenerationResult.Output.video(client.taskContent(id), null, duration)));
    }

    private static String imageData(String encoded) {
        byte[] bytes;
        try {
            bytes = Base64.getDecoder().decode(encoded);
        } catch (IllegalArgumentException invalid) {
            throw new BusinessException("贰春返回的图片内容无法解码");
        }
        String mime;
        if (bytes.length >= 8 && bytes[0] == (byte) 0x89 && bytes[1] == 'P' && bytes[2] == 'N' && bytes[3] == 'G') {
            mime = "image/png";
        } else if (bytes.length >= 3 && bytes[0] == (byte) 0xff && bytes[1] == (byte) 0xd8 && bytes[2] == (byte) 0xff) {
            mime = "image/jpeg";
        } else if (bytes.length >= 12 && bytes[0] == 'R' && bytes[1] == 'I' && bytes[8] == 'W' && bytes[9] == 'E') {
            mime = "image/webp";
        } else {
            throw new BusinessException("贰春返回的图片格式不受支持");
        }
        return "data:" + mime + ";base64," + encoded;
    }

    private static void validateTask(ObjectNode task, String id, YoukouContract contract) {
        requireId(id);
        if (!id.equals(task.path("id").asText()) || !"generation.task".equals(task.path("object").asText())
                || !contract.identity().equals(task.path("model").asText())
                || !contract.identity().equals(task.path("public_model_id").asText())
                || !contract.model().equals(task.path("public_model_name").asText())
                || !contract.kind().equals(task.path("output_type").asText())) {
            throw new BusinessException("贰春任务回执与所选模型不一致，请核对原任务");
        }
        // 合同固定使用这些路径，只保存编号即可恢复查询，不向响应提供的任意地址发送密钥。
        requirePath(task, "status_url", "/v1/tasks/" + id);
    }

    private static void requirePath(ObjectNode task, String key, String expected) {
        if (!expected.equals(task.path(key).asText())) {
            throw new BusinessException("贰春任务返回的结果地址无效，请核对原任务");
        }
    }

    private static void requireId(String id) {
        if (id == null || !id.matches("[1-9][0-9]*")) {
            throw new BusinessException("贰春任务编号无效，请核对任务详情");
        }
    }
}
