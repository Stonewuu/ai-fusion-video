package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.media.deletion.MediaReferenceRemoval;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class MediaReferenceRemovalTests {
    @Test
    void removesOnlyExplicitReferencesAndPreservesUnrelatedContent() {
        var json = new ObjectMapper();
        String id = "a7759143-98a9-4da3-a7ac-2346d806cdfa";
        var original = json.createObjectNode().put("resourceId", id).put("unrelatedId", id)
                .put("prompt", "参考 media://" + id + " 保留文字");
        original.putArray("mediaIds").add(id).add("different");
        original.putObject("inputMediaVersions").put(id, 1).put("different", 2);
        var patch = MediaReferenceRemoval.patch(original, id);
        assertThat(patch.path("resourceId").isNull()).isTrue();
        assertThat(patch.has("unrelatedId")).isFalse();
        assertThat(patch.path("prompt").asText()).isEqualTo("参考  保留文字");
        assertThat(patch.path("mediaIds")).hasSize(1);
        assertThat(patch.path("inputMediaVersions").has(id)).isFalse();
        assertThat(original.path("resourceId").asText()).isEqualTo(id);
    }
}
