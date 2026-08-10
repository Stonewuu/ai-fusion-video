package com.stonewu.fusion.entity.script;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.annotation.Version;
import com.stonewu.fusion.common.handler.JsonbTypeHandler;
import com.stonewu.fusion.common.BaseEntity;
import lombok.*;

/**
 * 剧本分场次实体
 * <p>
 * 对应数据库表：afv_script_scene_item
 * 存储剧本某集中的单个场次，包含场景信息、出场角色、对白列表等。
 */
@TableName(value = "afv_script_scene_item", autoResultMap = true)
@Data
@EqualsAndHashCode(callSuper = true)
@ToString(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ScriptSceneItem extends BaseEntity {

    /** 主键ID，自增 */
    @TableId(type = IdType.AUTO)
    private Long id;

    /** 所属分集ID */
    private Long episodeId;

    /** 所属剧本ID */
    private Long scriptId;

    /** 场次编号，如 "1-1"（第1集第1场） */
    private String sceneNumber;

    /** 场景标头，如 "内景 张三家客厅 夜" */
    private String sceneHeading;

    /** 场景地点 */
    private String location;

    /** 时间段：日/夜/黄昏/清晨 等 */
    private String timeOfDay;

    /** 内外景标识：内景/外景/内外景 */
    private String intExt;

    /** 出场角色名列表 JSON (List<String>) */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String characters;

    /** 出场角色资产ID列表 JSON (List<Long>) */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String characterAssetIds;

    /** 场景资产ID，关联 Asset */
    private Long sceneAssetId;

    /** 道具资产ID列表 JSON (List<Long>) */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String propAssetIds;

    /** 场景氛围/环境描述 */
    private String sceneDescription;

    /** 对白/动作元素列表 JSON (List<DialogueElement>)，按时间顺序排列 */
    @TableField(typeHandler = JsonbTypeHandler.class)
    private String dialogues;

    /** 排列顺序 */
    @Builder.Default
    private Integer sortOrder = 0;

    /** 状态：0-草稿 1-正常 */
    @Builder.Default
    private Integer status = 0;

    /** 乐观锁版本号 */
    @Version
    @Builder.Default
    private Integer version = 0;
}
