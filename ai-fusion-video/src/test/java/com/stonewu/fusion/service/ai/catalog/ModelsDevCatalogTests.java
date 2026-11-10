package com.stonewu.fusion.service.ai.catalog;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 搜索容忍名称格式与拼写差异，字段缺失不能变成“不支持”或虚构限额。 */
class ModelsDevCatalogTests {
    private final ObjectMapper json = new ObjectMapper();

    @Test
    void distinguishesUnknownFieldsAndReturnsExactFirstPartyBeforeRelay() throws Exception {
        var snapshot = ModelsDevCatalogService.parse(json.readTree("""
                {
                  "relay":{"name":"Relay","models":{"google/gemini-3.8-flash":{
                    "name":"Gemini 3.8 Flash","canonical_model_id":"google/gemini-3.8-flash",
                    "modalities":{"input":["text","image"],"output":["text"]}}}},
                  "google":{"name":"Google","models":{
                    "gemini-3.8-flash":{"name":"Gemini 3.8 Flash","canonical_model_id":"google/gemini-3.8-flash",
                      "limit":{"context":1048576,"output":65536},"reasoning":true,
                      "reasoning_options":[{"type":"effort","values":["low","medium","high"]}],
                      "modalities":{"input":["text","image","audio","pdf"],"output":["text"]}},
                    "gemini-3.1-flash-image":{"name":"Nano Banana 2","type":"image","reasoning":false,
                      "modalities":{"input":["text","image"],"output":["text","image"]}},
                    "gemini-2.5-flash":{"name":"Gemini 2.5 Flash","canonical_model_id":"google/gemini-2.5-flash",
                      "modalities":{"input":["text"],"output":["text"]}},
                    "tts":{"modalities":{"input":["text"],"output":["audio"]}},
                    "gemini-embedding-2":{"type":"embedding","modalities":{"output":["text"]}},
                    "gemini-3.5-transcribe":{"modalities":{"output":["text"]}},
                    "gemini-3.8-live":{"modalities":{"output":["audio"]}},
                    "deep-research-preview-04-2026":{"modalities":{"output":["text"]}}
                  }}
                }
                """));
        assertThat(snapshot.models()).hasSize(5);
        assertThat(ModelsDevMatcher.search(snapshot, "tts", 4, null).models()).hasSize(1);
        var matches = ModelsDevMatcher.search(snapshot, "gemini-3.8-flash", 1, null);
        assertThat(matches.models().getFirst().providerId()).isEqualTo("google");
        assertThat(matches.models().getFirst().contextWindow()).isEqualTo(1048576);
        assertThat(matches.models().get(1).contextWindow()).isNull();
        assertThat(matches.models().get(1).supportReasoning()).isNull();
        assertThat(ModelsDevMatcher.search(snapshot, "gemni 3.8 flash", 1, null).models().getFirst().modelId()).isEqualTo("gemini-3.8-flash");
        assertThat(ModelsDevMatcher.search(snapshot, "flash google 3.8", 1, null).total()).isEqualTo(2);
        assertThat(ModelsDevMatcher.search(snapshot, "Nano Banana", 2, null).models()).hasSize(1);
        assertThat(ModelsDevMatcher.search(snapshot, "Nano Banana", 2, null).models().getFirst().supportReasoning()).isFalse();
        assertThat(ModelsDevMatcher.search(snapshot, "not-a-model", 1, null).models()).isEmpty();
    }

    @Test
    void malformedCatalogDoesNotMasqueradeAsAnEmptySuccessfulSearch() throws Exception {
        assertThatThrownBy(() -> ModelsDevCatalogService.parse(json.readTree("{\"error\":\"upstream unavailable\"}")))
                .hasMessageContaining("暂时无法加载");
    }

    @Test
    void vendorFilterSearchesTheWholeCatalogBeforeLimitingResults() {
        var entries = new ArrayList<ModelsDevCatalogService.Entry>();
        for (int index = 0; index < 80; index++) {
            entries.add(new ModelsDevCatalogService.Entry("relay", "Relay", "model-" + index, "Model", null,
                    1, null, null, null, null, null, null, null, null, null));
        }
        entries.add(new ModelsDevCatalogService.Entry("official", "Official", "zz-model", "Model", null,
                1, null, null, null, null, null, null, null, null, null));
        var catalog = new ModelsDevCatalogService.Snapshot("2026-11-07", entries);
        assertThat(ModelsDevMatcher.search(catalog, "", 1, null).models()).hasSize(60);
        var filtered = ModelsDevMatcher.search(catalog, "", 1, "official");
        assertThat(filtered.models()).extracting(ModelsDevCatalogService.Entry::modelId).containsExactly("zz-model");
        assertThat(filtered.total()).isEqualTo(1);
    }
}
