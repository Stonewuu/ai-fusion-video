package com.stonewu.fusion.service.ai.text;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspacePayloadService;
import com.stonewu.fusion.service.resource.ScriptFields;

/**
 * 位置按 Unicode 字符计数，调用方直接使用返回的 nextOffset，不自行按字节或换行换算。
 */
public final class TextPages {

    private TextPages() {
    }

    public record Request(int offset, int limit, String contentVersion) {
        public static Request from(JsonNode input, Integer contextWindow) {
            int offset = input.hasNonNull("offset")
                    ? ScriptFields.integer(input.get("offset"), 0, Integer.MAX_VALUE, "offset") : 0;
            int budget = ToolTextBudget.pageChars(contextWindow);
            int limit = input.hasNonNull("limit")
                    ? Math.min(ScriptFields.integer(input.get("limit"), 1, ToolTextBudget.MAX_PAGE_CHARS, "limit"), budget)
                    : budget;
            JsonNode version = input.get("contentVersion");
            if (version != null && !version.isNull()
                    && (!version.isTextual() || !version.textValue().matches("[0-9a-f]{64}"))) {
                throw ScriptFields.invalid("contentVersion 必须使用上次读取返回的内容版本");
            }
            if (offset > 0 && (version == null || version.isNull())) {
                throw ScriptFields.invalid("继续读取时必须传入 contentVersion");
            }
            return new Request(offset, limit, version == null || version.isNull() ? null : version.textValue());
        }
    }

    public record Page(String content, String contentVersion, int offset, int endOffset,
                       int totalChars, Integer nextOffset, boolean hasMore) {
    }

    public static String version(String content) {
        return AgentWorkspacePayloadService.sha256(content);
    }

    public static Page read(String content, Request request) {
        String version = version(content);
        if (request.contentVersion() != null && !version.equals(request.contentVersion())) {
            throw ScriptFields.invalid("原文已变化，请从头读取并重新核对已处理范围");
        }
        int total = content.codePointCount(0, content.length());
        if (request.offset() > total) {
            throw ScriptFields.invalid("offset 超出正文范围");
        }
        int end = (int) Math.min((long) total, (long) request.offset() + request.limit());
        int startIndex = content.offsetByCodePoints(0, request.offset());
        int endIndex = content.offsetByCodePoints(startIndex, end - request.offset());
        return new Page(content.substring(startIndex, endIndex), version, request.offset(), end, total,
                end < total ? end : null, end < total);
    }
}
