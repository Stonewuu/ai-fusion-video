package com.stonewu.fusion.controller.storyboard.vo;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class StoryboardItemAssetsUpdateReqVOTests {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final Validator validator = Validation.buildDefaultValidatorFactory().getValidator();

    @Test
    void distinguishesExplicitEmptyArraysAndNullFromMissingFields() throws Exception {
        StoryboardItemAssetsUpdateReqVO explicitValues = objectMapper.readValue("""
                {
                  "characterIds": [],
                  "sceneAssetItemId": null,
                  "propIds": []
                }
                """, StoryboardItemAssetsUpdateReqVO.class);

        assertThat(explicitValues.isCharacterIdsPresent()).isTrue();
        assertThat(explicitValues.getCharacterIds()).isEmpty();
        assertThat(explicitValues.isSceneAssetItemIdPresent()).isTrue();
        assertThat(explicitValues.getSceneAssetItemId()).isNull();
        assertThat(explicitValues.isPropIdsPresent()).isTrue();
        assertThat(explicitValues.getPropIds()).isEmpty();
        assertThat(validator.validate(explicitValues)).isEmpty();

        StoryboardItemAssetsUpdateReqVO missingValues = objectMapper.readValue(
                "{}", StoryboardItemAssetsUpdateReqVO.class);

        assertThat(missingValues.isCharacterIdsPresent()).isFalse();
        assertThat(missingValues.isSceneAssetItemIdPresent()).isFalse();
        assertThat(missingValues.isPropIdsPresent()).isFalse();
        assertThat(validator.validate(missingValues)).isEmpty();
    }

    @Test
    void rejectsNullForMultiSelectFields() throws Exception {
        StoryboardItemAssetsUpdateReqVO request = objectMapper.readValue("""
                {
                  "characterIds": null,
                  "propIds": null
                }
                """, StoryboardItemAssetsUpdateReqVO.class);

        assertThat(validator.validate(request))
                .extracting(violation -> violation.getPropertyPath().toString())
                .containsExactlyInAnyOrder("characterIdsStateValid", "propIdsStateValid");
    }
}
