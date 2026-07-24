package com.stonewu.fusion.controller.storyboard;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardEpisodeBindReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardCreateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardEpisodeCreateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardEpisodeUpdateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardFrameUpdateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardItemAssetsUpdateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardItemCreateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardItemSortReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardItemUpdateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardSceneCreateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardSceneUpdateReqVO;
import com.stonewu.fusion.controller.storyboard.vo.StoryboardUpdateReqVO;
import com.stonewu.fusion.convert.storyboard.StoryboardConvert;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import com.stonewu.fusion.service.storyboard.VideoComposeService;
import com.stonewu.fusion.service.storyboard.dto.StoryboardItemAssetsPatch;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.security.SecurityUtils.requireCurrentUserId;

/**
 * 分镜脚本 Controller
 */
@Tag(name = "分镜管理")
@RestController
@RequestMapping("/api/storyboard")
@RequiredArgsConstructor
public class StoryboardController {

    private final StoryboardService storyboardService;
    private final VideoComposeService videoComposeService;

    // ========== 分镜脚本 ==========

    @Operation(summary = "获取分镜详情")
    @GetMapping("/{id}")
    public CommonResult<Storyboard> get(@PathVariable Long id) {
        return CommonResult.success(storyboardService.getById(id));
    }

    @Operation(summary = "按项目查询分镜列表")
    @GetMapping("/list")
    public CommonResult<List<Storyboard>> list(@RequestParam Long projectId) {
        return CommonResult.success(storyboardService.listByProject(projectId));
    }

    @Operation(summary = "创建分镜")
    @PostMapping
    public CommonResult<Storyboard> create(@Valid @RequestBody StoryboardCreateReqVO reqVO) {
        Storyboard storyboard = StoryboardConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(storyboardService.create(storyboard));
    }

    @Operation(summary = "更新分镜")
    @PutMapping
    public CommonResult<Storyboard> update(@Valid @RequestBody StoryboardUpdateReqVO reqVO) {
        Storyboard storyboard = StoryboardConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(storyboardService.update(storyboard));
    }

    @Operation(summary = "删除分镜")
    @DeleteMapping("/{id}")
    public CommonResult<Boolean> delete(@PathVariable Long id) {
        storyboardService.delete(id);
        return CommonResult.success(true);
    }

    // ========== 分镜集 ==========

    @Operation(summary = "获取分镜集列表")
    @GetMapping("/{storyboardId}/episodes")
    public CommonResult<List<StoryboardEpisode>> listEpisodes(@PathVariable Long storyboardId) {
        return CommonResult.success(storyboardService.listEpisodes(storyboardId));
    }

    @Operation(summary = "获取分镜集详情")
    @GetMapping("/episode/{id}")
    public CommonResult<StoryboardEpisode> getEpisode(@PathVariable Long id) {
        return CommonResult.success(storyboardService.getEpisodeById(id));
    }

    @Operation(summary = "创建分镜集")
    @PostMapping("/episode")
    public CommonResult<StoryboardEpisode> createEpisode(@Valid @RequestBody StoryboardEpisodeCreateReqVO reqVO) {
        StoryboardEpisode episode = StoryboardConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(storyboardService.createEpisode(episode));
    }

    @Operation(summary = "更新分镜集")
    @PutMapping("/episode")
    public CommonResult<StoryboardEpisode> updateEpisode(@Valid @RequestBody StoryboardEpisodeUpdateReqVO reqVO) {
        StoryboardEpisode episode = StoryboardConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(storyboardService.updateEpisode(episode));
    }

    @Operation(summary = "删除分镜集")
    @DeleteMapping("/episode/{id}")
    public CommonResult<Boolean> deleteEpisode(@PathVariable Long id) {
        storyboardService.deleteEpisode(id);
        return CommonResult.success(true);
    }

    /**
     * 绑定分镜集和剧本分集。
     *
     * @param id    分镜集ID
     * @param reqVO 绑定请求
     * @return 绑定后的分镜集
     */
    @Operation(summary = "绑定分镜集和剧本分集")
    @PutMapping("/episode/{id}/bindScriptEpisode")
    public CommonResult<StoryboardEpisode> bindScriptEpisode(@PathVariable Long id,
                                                             @Valid @RequestBody StoryboardEpisodeBindReqVO reqVO) {
        return CommonResult.success(storyboardService.bindScriptEpisode(id, reqVO.getScriptEpisodeId()));
    }

    /**
     * 清空分镜集下的场次和镜头。
     *
     * @param id 分镜集ID
     * @return 是否清空成功
     */
    @Operation(summary = "清空分镜集内容")
    @PostMapping("/episode/{id}/clearContent")
    public CommonResult<Boolean> clearEpisodeContent(@PathVariable Long id) {
        storyboardService.clearEpisodeContent(id);
        return CommonResult.success(true);
    }

    @Operation(summary = "提交本集合成视频任务（异步）")
    @PostMapping("/episode/{id}/compose-video")
    public CommonResult<String> composeEpisodeVideo(@PathVariable Long id) {
        Long userId = requireCurrentUserId();
        return CommonResult.success(videoComposeService.submitCompose(id, userId));
    }

    // ========== 分镜场次 ==========

    @Operation(summary = "获取分镜场次列表（按集）")
    @GetMapping("/episode/{episodeId}/scenes")
    public CommonResult<List<StoryboardScene>> listScenesByEpisode(@PathVariable Long episodeId) {
        return CommonResult.success(storyboardService.listScenesByEpisode(episodeId));
    }

    @Operation(summary = "获取分镜场次列表（按分镜）")
    @GetMapping("/{storyboardId}/scenes")
    public CommonResult<List<StoryboardScene>> listScenesByStoryboard(@PathVariable Long storyboardId) {
        return CommonResult.success(storyboardService.listScenesByStoryboard(storyboardId));
    }

    @Operation(summary = "获取分镜场次详情")
    @GetMapping("/scene/{id}")
    public CommonResult<StoryboardScene> getScene(@PathVariable Long id) {
        return CommonResult.success(storyboardService.getSceneById(id));
    }

    @Operation(summary = "创建分镜场次")
    @PostMapping("/scene")
    public CommonResult<StoryboardScene> createScene(@Valid @RequestBody StoryboardSceneCreateReqVO reqVO) {
        StoryboardScene scene = StoryboardConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(storyboardService.createScene(scene));
    }

    @Operation(summary = "更新分镜场次")
    @PutMapping("/scene")
    public CommonResult<StoryboardScene> updateScene(@Valid @RequestBody StoryboardSceneUpdateReqVO reqVO) {
        StoryboardScene scene = StoryboardConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(storyboardService.updateScene(scene));
    }

    @Operation(summary = "删除分镜场次")
    @DeleteMapping("/scene/{id}")
    public CommonResult<Boolean> deleteScene(@PathVariable Long id) {
        storyboardService.deleteScene(id);
        return CommonResult.success(true);
    }

    // ========== 分镜条目 ==========

    @Operation(summary = "获取分镜条目列表（按分镜）")
    @GetMapping("/{storyboardId}/items")
    public CommonResult<List<StoryboardItem>> listItems(@PathVariable Long storyboardId) {
        return CommonResult.success(storyboardService.listItems(storyboardId));
    }

    @Operation(summary = "获取分镜条目列表（按场次）")
    @GetMapping("/scene/{sceneId}/items")
    public CommonResult<List<StoryboardItem>> listItemsByScene(@PathVariable Long sceneId) {
        return CommonResult.success(storyboardService.listItemsByScene(sceneId));
    }

    @Operation(summary = "获取分镜条目详情")
    @GetMapping("/item/{id}")
    public CommonResult<StoryboardItem> getItem(@PathVariable Long id) {
        return CommonResult.success(storyboardService.getItemById(id));
    }

    @Operation(summary = "创建分镜条目")
    @PostMapping("/item")
    public CommonResult<StoryboardItem> createItem(@Valid @RequestBody StoryboardItemCreateReqVO reqVO) {
        StoryboardItem item = StoryboardConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(storyboardService.createItem(item));
    }

    @Operation(summary = "更新分镜条目")
    @PutMapping("/item")
    public CommonResult<StoryboardItem> updateItem(@Valid @RequestBody StoryboardItemUpdateReqVO reqVO) {
        StoryboardItem item = StoryboardConvert.INSTANCE.convert(reqVO);
        return CommonResult.success(storyboardService.updateItem(item));
    }

    @Operation(summary = "局部更新分镜条目资产关联")
    @PatchMapping("/item/{id}/assets")
    public CommonResult<StoryboardItem> updateItemAssets(
            @PathVariable Long id,
            @Valid @RequestBody StoryboardItemAssetsUpdateReqVO reqVO) {
        StoryboardItemAssetsPatch patch = new StoryboardItemAssetsPatch(
                reqVO.isCharacterIdsPresent(),
                reqVO.getCharacterIds(),
                reqVO.isSceneAssetItemIdPresent(),
                reqVO.getSceneAssetItemId(),
                reqVO.isPropIdsPresent(),
                reqVO.getPropIds()
        );
        return CommonResult.success(storyboardService.updateItemAssets(id, patch));
    }

    /**
     * 更新分镜条目的首帧或尾帧参考图。
     *
     * @param id    分镜条目ID
     * @param reqVO 首尾帧更新请求
     * @return 更新后的分镜条目
     */
    @Operation(summary = "更新分镜条目首尾帧")
    @PutMapping("/item/{id}/updateFrame")
    public CommonResult<StoryboardItem> updateItemFrame(@PathVariable Long id,
                                                       @Valid @RequestBody StoryboardFrameUpdateReqVO reqVO) {
        return CommonResult.success(storyboardService.updateItemFrame(
                id,
                reqVO.getFrameType(),
                reqVO.getImageUrl(),
                reqVO.getPrompt()
        ));
    }

    @Operation(summary = "删除分镜条目")
    @DeleteMapping("/item/{id}")
    public CommonResult<Boolean> deleteItem(@PathVariable Long id) {
        storyboardService.deleteItem(id);
        return CommonResult.success(true);
    }

    @Operation(summary = "批量创建分镜条目")
    @PostMapping("/{storyboardId}/items/batch")
    public CommonResult<Boolean> batchCreate(@PathVariable Long storyboardId,
                                             @RequestBody List<StoryboardItemCreateReqVO> reqVOList) {
        List<StoryboardItem> items = StoryboardConvert.INSTANCE.convertCreateList(reqVOList);
        items.forEach(item -> item.setStoryboardId(storyboardId));
        storyboardService.batchCreateItems(items);
        return CommonResult.success(true);
    }

    @Operation(summary = "批量更新分镜条目排序")
    @PostMapping("/items/batch-sort")
    public CommonResult<Boolean> batchUpdateSort(@Valid @RequestBody StoryboardItemSortReqVO reqVO) {
        storyboardService.batchUpdateItemSort(reqVO.getIds());
        return CommonResult.success(true);
    }
}
