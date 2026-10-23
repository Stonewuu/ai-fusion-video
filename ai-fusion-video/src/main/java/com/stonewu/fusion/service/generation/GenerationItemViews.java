package com.stonewu.fusion.service.generation;

import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.service.media.MediaReference;
import org.springframework.beans.BeanUtils;

/**
 * 新任务的临时供应商 URL 只供工作者使用，浏览器只看到已登记结果。历史任务保留原始访问方式。
 */
public final class GenerationItemViews {
    private GenerationItemViews() {
    }

    public static ImageItem project(ImageItem item, boolean managedTask, int taskStatus) {
        ImageItem result = new ImageItem();
        BeanUtils.copyProperties(item, result);
        if (!managedTask) {
            return result;
        }
        result.setImageUrl(ready(item.getImageUrl()));
        result.setThumbnailUrl(ready(item.getThumbnailUrl()));
        if (Integer.valueOf(1).equals(item.getStatus()) && result.getImageUrl() == null) {
            result.setStatus(taskStatus >= 2 ? 2 : 0);
            if (taskStatus >= 2) {
                result.setErrorMsg("图片未完成媒体登记，请查看任务错误");
            }
        }
        return result;
    }

    public static VideoItem project(VideoItem item, boolean managedTask, int taskStatus) {
        VideoItem result = new VideoItem();
        BeanUtils.copyProperties(item, result);
        if (!managedTask) {
            return result;
        }
        result.setVideoUrl(ready(item.getVideoUrl()));
        result.setCoverUrl(ready(item.getCoverUrl()));
        result.setFirstFrameUrl(ready(item.getFirstFrameUrl()));
        result.setLastFrameUrl(ready(item.getLastFrameUrl()));
        if (Integer.valueOf(1).equals(item.getStatus()) && result.getVideoUrl() == null) {
            result.setStatus(taskStatus >= 2 ? 2 : 0);
            if (taskStatus >= 2) {
                result.setErrorMsg("视频未完成媒体登记，请查看任务错误");
            }
        }
        return result;
    }

    private static String ready(String value) {
        return MediaReference.managed(value) ? value : null;
    }
}
