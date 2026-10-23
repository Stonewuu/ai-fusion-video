package com.stonewu.fusion.service.ai.provider;

/** 本地 HTTP 测试的协议分片，首段和结束段由测试服务器分别 flush。 */
public final class StreamingProviderFixtures {
    private StreamingProviderFixtures() { }

    public static String first(String protocol) {
        return switch (protocol) {
            case "openai" -> sse("""
                    {"id":"test-id","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"雨落"}}]}
                    """);
            case "anthropic" -> """
                    event: message_start
                    data: {"type":"message_start","message":{"id":"test-id","type":"message","model":"test-text-model","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}

                    event: content_block_start
                    data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

                    event: content_block_delta
                    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"雨落"}}

                    """;
            case "gemini", "vertex" -> sse("""
                    {"responseId":"test-id","candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"雨落"}]}}]}
                    """);
            case "dashscope" -> sse("""
                    {"request_id":"test-id","output":{"choices":[{"message":{"role":"assistant","content":"雨落"},"finish_reason":"null"}]}}
                    """);
            case "ollama" -> """
                    {"model":"test-text-model","created_at":"2026-10-15T00:00:00Z","message":{"role":"assistant","content":"雨落"},"done":false}
                    """;
            case "responses" -> """
                    event: response.output_text.delta
                    data: {"type":"response.output_text.delta","item_id":"msg-test","output_index":0,"content_index":0,"delta":"雨落","sequence_number":1}

                    """;
            default -> throw new IllegalArgumentException(protocol);
        };
    }

    public static String last(String protocol) {
        return switch (protocol) {
            case "openai" -> sse("""
                    {"id":"test-id","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"古城。"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
                    """) + "data: [DONE]\n\n";
            case "anthropic" -> """
                    event: content_block_delta
                    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"古城。"}}

                    event: content_block_stop
                    data: {"type":"content_block_stop","index":0}

                    event: message_delta
                    data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

                    event: message_stop
                    data: {"type":"message_stop"}

                    """;
            case "gemini", "vertex" -> sse("""
                    {"responseId":"test-id","candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"古城。"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}
                    """);
            case "dashscope" -> sse("""
                    {"request_id":"test-id","output":{"choices":[{"message":{"role":"assistant","content":"古城。"},"finish_reason":"stop"}]},"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}
                    """);
            case "ollama" -> """
                    {"model":"test-text-model","created_at":"2026-10-15T00:00:00Z","message":{"role":"assistant","content":"古城。"},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":5}
                    """;
            case "responses" -> """
                    event: response.output_text.delta
                    data: {"type":"response.output_text.delta","item_id":"msg-test","output_index":0,"content_index":0,"delta":"古城。","sequence_number":2}

                    event: response.completed
                    data: {"type":"response.completed","sequence_number":3,"response":{"id":"resp-test","status":"completed","model":"test-text-model","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}

                    """;
            default -> throw new IllegalArgumentException(protocol);
        };
    }

    public static String sse(String json) {
        return "data: " + json.strip() + "\n\n";
    }
}
