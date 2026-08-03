package com.stonewu.fusion.controller.storyboard.vo;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonSetter;
import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.AssertTrue;
import jakarta.validation.constraints.NotNull;

import java.util.List;

/**
 * 分镜条目资产关联局部更新请求。
 *
 * <p>字段缺省表示不修改；角色、道具传空数组表示清空；场景显式传 null 表示清空。</p>
 */
@Schema(description = "分镜条目资产关联局部更新请求")
public class StoryboardItemAssetsUpdateReqVO {

    @Schema(description = "角色子资产 ID 列表；字段缺省表示不修改，空数组表示清空")
    private List<@NotNull(message = "角色子资产ID不能为空") Long> characterIds;

    @Schema(description = "场景子资产 ID；字段缺省表示不修改，显式 null 表示清空")
    private Long sceneAssetItemId;

    @Schema(description = "道具子资产 ID 列表；字段缺省表示不修改，空数组表示清空")
    private List<@NotNull(message = "道具子资产ID不能为空") Long> propIds;

    private boolean characterIdsPresent;
    private boolean sceneAssetItemIdPresent;
    private boolean propIdsPresent;

    public List<Long> getCharacterIds() {
        return characterIds;
    }

    @JsonSetter("characterIds")
    public void setCharacterIds(List<Long> characterIds) {
        this.characterIds = characterIds;
        this.characterIdsPresent = true;
    }

    public Long getSceneAssetItemId() {
        return sceneAssetItemId;
    }

    @JsonSetter("sceneAssetItemId")
    public void setSceneAssetItemId(Long sceneAssetItemId) {
        this.sceneAssetItemId = sceneAssetItemId;
        this.sceneAssetItemIdPresent = true;
    }

    public List<Long> getPropIds() {
        return propIds;
    }

    @JsonSetter("propIds")
    public void setPropIds(List<Long> propIds) {
        this.propIds = propIds;
        this.propIdsPresent = true;
    }

    @JsonIgnore
    @Schema(hidden = true)
    public boolean isCharacterIdsPresent() {
        return characterIdsPresent;
    }

    @JsonIgnore
    @Schema(hidden = true)
    public boolean isSceneAssetItemIdPresent() {
        return sceneAssetItemIdPresent;
    }

    @JsonIgnore
    @Schema(hidden = true)
    public boolean isPropIdsPresent() {
        return propIdsPresent;
    }

    @AssertTrue(message = "角色关联清空时请传空数组，不能传 null")
    @JsonIgnore
    @Schema(hidden = true)
    public boolean isCharacterIdsStateValid() {
        return !characterIdsPresent || characterIds != null;
    }

    @AssertTrue(message = "道具关联清空时请传空数组，不能传 null")
    @JsonIgnore
    @Schema(hidden = true)
    public boolean isPropIdsStateValid() {
        return !propIdsPresent || propIds != null;
    }
}
