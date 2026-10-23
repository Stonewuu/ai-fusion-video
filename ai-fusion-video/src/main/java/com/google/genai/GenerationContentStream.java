package com.google.genai;

import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.FluxSink;
import reactor.core.scheduler.Schedulers;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * 复用 Google GenAI 1.45 的请求及响应转换，只替换会预读下一片的 ResponseStream 迭代器。
 * 每收到一个完整 SSE 事件立即交付；取消订阅关闭原连接，不重新提交模型请求。
 */
public final class GenerationContentStream {
    private GenerationContentStream() {
    }

    public static Flux<GenerateContentResponse> generate(
            GenerationHttpApiClient client, String model, List<Content> contents, GenerateContentConfig config) {
        return Flux.<GenerateContentResponse>create(sink -> {
            Models models = new Models(client);
            var request = models.buildRequestForPrivateGenerateContentStream(model, contents, config);
            try (ApiResponse response = client.request("post", request.path(), request.body(), request.httpOptions())) {
                sink.onCancel(response::close);
                try (var reader = new BufferedReader(
                        new InputStreamReader(response.getBody().byteStream(), StandardCharsets.UTF_8))) {
                    StringBuilder data = new StringBuilder();
                    String line;
                    while (!sink.isCancelled() && (line = reader.readLine()) != null) {
                        if (line.isEmpty()) {
                            if (!data.isEmpty()) {
                                emit(sink, models, client.vertexAI(), data.toString());
                                data.setLength(0);
                            }
                        } else if (line.startsWith("data:")) {
                            if (!data.isEmpty()) {
                                data.append('\n');
                            }
                            String value = line.substring(5);
                            data.append(value.startsWith(" ") ? value.substring(1) : value);
                        }
                    }
                    if (!sink.isCancelled() && !data.isEmpty()) {
                        throw new IllegalStateException("Gemini 流在事件结束前中断");
                    }
                    sink.complete();
                }
            } catch (Exception failure) {
                if (!sink.isCancelled()) {
                    sink.error(failure);
                }
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    private static void emit(FluxSink<GenerateContentResponse> sink, Models models, boolean vertex, String data) {
        var node = JsonSerializable.stringToJsonNode(data);
        if (node.has("error")) {
            throw new IllegalStateException("Gemini 在流式响应中返回错误");
        }
        var converted = vertex ? models.generateContentResponseFromVertex(node, null, node)
                : models.generateContentResponseFromMldev(node, null, node);
        sink.next(JsonSerializable.fromJsonNode(converted, GenerateContentResponse.class));
    }
}
