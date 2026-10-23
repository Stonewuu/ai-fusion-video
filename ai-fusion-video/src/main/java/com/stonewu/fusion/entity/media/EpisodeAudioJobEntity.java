package com.stonewu.fusion.entity.media;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("afv_episode_audio_job")
public class EpisodeAudioJobEntity {
    /**
     * 分集合成任务标识
     */
    @TableId(type = IdType.INPUT)
    private String jobId;

    /**
     * 冻结的音频服务配置编号
     */
    private Long apiConfigId;

    /**
     * 音频服务连接摘要
     */
    private String apiConfigHash;

    /**
     * 是否需要生成音效
     */
    private Boolean sfxEnabled;

    /**
     * 是否预留或持有远端音频并发额度
     */
    private Boolean permitHeld;

    /**
     * 已确认的配乐终态响应，禁止直接暴露给客户端
     */
    private String musicResultJson;

    /**
     * 不可变配乐响应摘要
     */
    private String musicResultHash;

    /**
     * 已确认的音效终态响应，禁止直接暴露给客户端
     */
    private String sfxResultJson;

    /**
     * 不可变音效响应摘要
     */
    private String sfxResultHash;

    /**
     * 连续查询失败次数
     */
    private Integer consecutiveErrors;

    /**
     * 已人工确认远端结束的步骤，保留原提交和响应
     */
    private String resolvedStepsJson;

    /**
     * 最近一次人工核实的依据、原身份和动作摘要
     */
    private String reviewJson;

    /**
     * 升级前缺少终态证据的本地任务状态
     */
    private String legacyState;
}
