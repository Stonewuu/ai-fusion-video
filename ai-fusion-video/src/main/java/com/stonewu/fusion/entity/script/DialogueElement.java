package com.stonewu.fusion.entity.script;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 对白/动作元素（嵌套 JSON 对象，非数据库实体）
 * <p>
 * 用于 SceneItem.dialogues 字段内的 JSON 解析。
 * 表示剧本中的一行对白、动作描写、画外音或镜头指令。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DialogueElement {

    /** 类型：1-对白 2-动作描写(▲) 3-画外音(V.O.) 4-镜头指令(【】) 5-环境描写 */
    private String type;

    /** 角色名（type=1 对白 和 type=3 画外音时必填） */
    private String character;

    /** 文本内容 */
    private String content;

    /** 括号说明，如"（低声）""（愤怒地）" */
    private String parenthetical;

    /** 排列顺序（在场次内的顺序） */
    private Integer sortOrder;
}
