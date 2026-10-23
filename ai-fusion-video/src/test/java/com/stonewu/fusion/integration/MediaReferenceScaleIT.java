package com.stonewu.fusion.integration;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;

import static org.assertj.core.api.Assertions.assertThat;

/** 大历史量验收只使用独立 Testcontainers 数据和真实保留视图。 */
@Testcontainers
class MediaReferenceScaleIT extends MediaResourceTestContext {
    @Test
    @Timeout(300)
    void mixedHistoryRetainsReferencedMediaAndReportsActualLookupCosts() throws Exception {
        String messageMedia = upload(id(), png, "历史消息.png", null, USER).id();
        String commandMedia = upload(id(), png, "历史命令.png", null, USER).id();
        String checkpointMedia = upload(id(), png, "命名版本.png", null, USER).id();
        String unreferencedMedia = upload(id(), png, "没有引用.png", null, USER).id();
        String canvas = id();
        jdbc.update("INSERT INTO afv_canvas(id,title,owner_type,owner_id,created_by) VALUES(?,'大历史验收',1,71,71)", canvas);
        String body = "雨夜，阿青来到灯塔前，老船长正在等一封没有署名的信。镜头缓慢推进，保留蓝色雨衣与人物对白。".repeat(24);
        String plainJson = mapper.writeValueAsString(Map.of("description", body));
        String commandJson = mapper.writeValueAsString(Map.of("description", body, "mediaId", commandMedia));
        String checkpointJson = mapper.writeValueAsString(Map.of("description", body,
                "resources", List.of(Map.of("data", Map.of("imageUrl", "media://" + checkpointMedia)))));
        insert(20_000, "INSERT INTO afv_agent_message(conversation_id,role,content,message_order) VALUES(?,'assistant',?,?)",
                index -> new Object[]{"scale-history", body + (index == 19_999 ? " [图片](media://" + messageMedia + ")" : ""), index});
        insert(10_000, """
                INSERT INTO afv_canvas_operation(id,canvas_id,actor_user_id,request_id,request_hash,origin,
                    committed_revision,request_json,result_json) VALUES(?,?,71,?,?,'user',?,?,'{}')
                """, index -> new Object[]{id(), canvas, "scale-" + index, "a".repeat(64), index + 1,
                    index == 9_999 ? commandJson : plainJson});
        insert(1_000, """
                INSERT INTO afv_canvas_revision(id,canvas_id,title,revision,sequence_no,snapshot_json,created_by)
                VALUES(?,?,'历史命名版本',?,?,?,71)
                """, index -> new Object[]{id(), canvas, index + 1, index + 1, index == 999 ? checkpointJson : plainJson});

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("database", MYSQL.getDockerImageName());
        report.put("agentMessages", 20_000);
        report.put("canvasCommands", 10_000);
        report.put("namedVersions", 1_000);
        report.put("bodyCharactersPerRow", body.length());
        List<Map<String, Object>> measurements = new ArrayList<>();
        report.put("measurements", measurements);
        jdbc.setQueryTimeout(60);
        try {
            for (int round = 0; round < 3; round++) {
                lookup(messageMedia, "AGENT_MESSAGE", round, measurements);
                lookup(commandMedia, "CANVAS_COMMAND", round, measurements);
                lookup(checkpointMedia, "CANVAS_REVISION", round, measurements);
                lookup(unreferencedMedia, null, round, measurements);
            }
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_message WHERE conversation_id='scale-history'", Integer.class))
                    .isEqualTo(20_000);
        } finally {
            jdbc.setQueryTimeout(0);
            Path reportPath = Path.of("target", "g-media-reference-scale-results.json");
            Files.writeString(reportPath, mapper.writerWithDefaultPrettyPrinter().writeValueAsString(report));
            System.out.println("媒体保留查询规模验收记录: " + reportPath.toAbsolutePath());
        }
    }

    private void lookup(String mediaId, String expectedKind, int round, List<Map<String, Object>> measurements) {
        long started = System.nanoTime();
        List<String> found = repository.retentionReference(mediaId).stream().toList();
        double elapsedMillis = (System.nanoTime() - started) / 1_000_000.0;
        measurements.add(Map.of("round", round, "expectedKind", expectedKind == null ? "UNREFERENCED" : expectedKind,
                "milliseconds", elapsedMillis, "actualKinds", found));
        if (expectedKind == null) assertThat(found).isEmpty();
        else assertThat(found).containsExactly(expectedKind);
    }

    private void insert(int count, String sql, IntFunction<Object[]> values) {
        for (int start = 0; start < count; start += 500) {
            List<Object[]> batch = new ArrayList<>();
            for (int index = start; index < Math.min(start + 500, count); index++) batch.add(values.apply(index));
            // 数据准备按批提交，避免逐行 fsync 的时间掩盖真正需要测量的查询成本。
            context.getBean(TransactionTemplate.class).executeWithoutResult(status -> jdbc.batchUpdate(sql, batch));
        }
    }
}
