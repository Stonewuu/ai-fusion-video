package com.stonewu.fusion.service.ai.google;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.generation.audio.AudioTaskData;
import com.stonewu.fusion.service.generation.video.strategy.google.GoogleVideoProtocol;
import java.util.ArrayList;
import java.util.List;

public final class GoogleAudioProtocol {
    private GoogleAudioProtocol() {
    }

    public record Request(String operation, JSONObject body) {
    }

    public record Audio(String data, String mime) {
    }

    public record Output(String text, List<Audio> audio) {
    }

    public static final class InvalidOutput extends BusinessException {
        public InvalidOutput(String message) {
            super(message);
        }
    }

    public static boolean music(String code) {
        return code.startsWith("lyria-");
    }

    public static Request request(AiModel model, AudioTaskData.Input input,
                                  List<String> media, JSONObject config, boolean developer) {
        boolean modernSpeech = model.getCode().startsWith("gemini-3.8-");
        if (model.getCode().startsWith("lyria-3") || developer && modernSpeech) {
            var content = new JSONArray().put(new JSONObject().set("type", "text").set("text", input.text()));
            for (String source : media) {
                content.add(GoogleVideoProtocol.interactionMedia(source, "image"));
            }
            var body = new JSONObject().set("model", model.getCode())
                    .set("input", GoogleInteractionsProtocol.input(content))
                    .set("store", false).set("stream", false);
            if (!music(model.getCode())) {
                body.set("response_format", new JSONObject().set("type", "audio"))
                        .set("generation_config", new JSONObject().set("speech_config",
                                new JSONArray().put(new JSONObject().set("voice", voice(input, config)))));
            }
            return new Request("interactions", body);
        }
        if ("lyria-002".equals(model.getCode())) {
            return new Request("predict", new JSONObject().set("instances",
                    new JSONArray().put(new JSONObject().set("prompt", input.text())))
                    .set("parameters", new JSONObject().set("sample_count", 1)));
        }
        var voiceConfig = modernSpeech ? new JSONObject().set("voice", voice(input, config))
                : new JSONObject().set("prebuiltVoiceConfig", new JSONObject().set("voiceName", voice(input, config)));
        var generation = new JSONObject().set("responseModalities", List.of("AUDIO"))
                .set("speechConfig", new JSONObject().set("voiceConfig", voiceConfig));
        var content = new JSONObject().set("role", "user").set("parts",
                new JSONArray().put(new JSONObject().set("text", input.text())));
        return new Request("generateContent", new JSONObject().set("contents", new JSONArray().put(content))
                .set("generationConfig", generation));
    }

    private static String voice(AudioTaskData.Input input, JSONObject config) {
        return StrUtil.blankToDefault(input.voice(), config.getStr("voice", "Kore"));
    }

    public static Output output(JSONObject response, String operation) {
        var text = new StringBuilder();
        var audio = new ArrayList<Audio>();
        if ("interactions".equals(operation)) {
            if (!"completed".equals(response.getStr("status"))) {
                throw new InvalidOutput("Google 尚未返回完整音频，请稍后重试");
            }
            var steps = response.getJSONArray("steps");
            if (steps != null) {
                for (JSONObject step : steps.toList(JSONObject.class)) {
                    if ("model_output".equals(step.getStr("type")) && step.getJSONArray("content") != null) {
                        for (JSONObject content : step.getJSONArray("content").toList(JSONObject.class)) {
                            if ("text".equals(content.getStr("type"))) {
                                text.append(content.getStr("text", ""));
                            } else if ("audio".equals(content.getStr("type"))) {
                                audio.add(new Audio(content.getStr("data"), content.getStr("mime_type")));
                            }
                        }
                    }
                }
            }
        } else if ("predict".equals(operation)) {
            var predictions = response.getJSONArray("predictions");
            if (predictions != null) {
                for (JSONObject item : predictions.toList(JSONObject.class)) {
                    if (item.containsKey("bytesBase64Encoded")) {
                        // Lyria 002 返回固定 WAV；官方文档另列出 audioContent 格式。
                        audio.add(new Audio(item.getStr("bytesBase64Encoded"), "audio/wav"));
                    } else if (item.containsKey("audioContent") && item.containsKey("mimeType")) {
                        audio.add(new Audio(item.getStr("audioContent"), item.getStr("mimeType")));
                    }
                }
            }
        } else if ("generateContent".equals(operation)) {
            var candidates = response.getJSONArray("candidates");
            if (candidates != null) {
                for (JSONObject candidate : candidates.toList(JSONObject.class)) {
                    var parts = candidate.getByPath("content.parts", JSONArray.class);
                    if (parts == null) {
                        continue;
                    }
                    for (JSONObject part : parts.toList(JSONObject.class)) {
                        if (part.getBool("thought", false)) {
                            continue;
                        }
                        text.append(part.getStr("text", ""));
                        var inline = part.getJSONObject("inlineData");
                        if (inline != null && inline.getStr("mimeType", "").startsWith("audio/")) {
                            audio.add(new Audio(inline.getStr("data"), inline.getStr("mimeType")));
                        }
                    }
                }
            }
        } else {
            throw new InvalidOutput("音频响应格式无法识别");
        }
        if (audio.isEmpty() || audio.stream().anyMatch(item -> StrUtil.isBlank(item.data()) || StrUtil.isBlank(item.mime()))) {
            throw new InvalidOutput("Google 未返回音频，请调整输入后重试");
        }
        return new Output(text.toString(), List.copyOf(audio));
    }
}
