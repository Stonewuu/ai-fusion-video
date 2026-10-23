package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.media.lifecycle.MediaReferenceIndex;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;
import static org.assertj.core.api.Assertions.*;

class MediaReferenceIndexTests {
    private final ObjectMapper mapper=new ObjectMapper();
    private final String first="b3649062-3f3c-4f9c-b7de-a13a215d177a",second="a27c87c9-5b53-42cc-9bb8-c60c98806886";
    @Test void extractsExplicitLinksTypedIdsAndGenerationKeysWithoutGuessingUnrelatedIds() throws Exception {
        String payload=mapper.writeValueAsString(Map.of("content","[图](media://"+first+")", "nodes",List.of(Map.of("mediaId",first)),
                "inputMediaVersions",Map.of(second,1),"foreignModelId","df5762b0-0ef1-47e1-b03b-d9a2a42af3c6"));
        assertThat(mapper.readTree(MediaReferenceIndex.extract(payload,mapper))).isEqualTo(mapper.valueToTree(List.of("media://"+second,"media://"+first)));
    }
    @Test void malformedDocumentsCannotBecomeAnEmptyIndexAndInvalidUuidSuffixesAreNotAccepted() {
        assertThatThrownBy(() -> MediaReferenceIndex.extract("broken",mapper)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> MediaReferenceIndex.extract("{} {\"mediaId\":\""+first+"\"}",mapper)).isInstanceOf(IllegalArgumentException.class);
        assertThat(MediaReferenceIndex.extract("{\"content\":\"media://"+first+"-extra\"}",mapper)).isEqualTo("[]");
    }
}
