package com.stonewu.fusion.service.generation.image;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.generation.job.GenerationDeletionGuard;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.mapper.generation.ImageTaskMapper;
import com.stonewu.fusion.mapper.generation.ImageItemMapper;
import org.junit.jupiter.api.Test;
import com.stonewu.fusion.service.media.GenerationMediaLinks;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class ImageGenerationDeletionTests {
    private static final CanvasActor ACTOR = new CanvasActor(2L, null);
    private final ImageTaskMapper tasks = mock(ImageTaskMapper.class);
    private final ImageItemMapper items = mock(ImageItemMapper.class);
    private final GenerationDeletionGuard guard = mock(GenerationDeletionGuard.class);
    private final ImageGenerationService service = new ImageGenerationService(tasks, items, guard, mock(GenerationMediaLinks.class));

    @ParameterizedTest
    @ValueSource(ints = {0, 1})
    void rejectsUnfinishedTask(int status) {
        task(status);
        assertThatThrownBy(() -> service.delete(7L, ACTOR))
                .isInstanceOf(BusinessException.class).hasMessageContaining("生成结束");
        verifyNoInteractions(items);
        verify(guard, never()).requireTerminal(GenerationKind.IMAGE, 7L);
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 3, 4, 6})
    void deletesFinishedTaskAndItsItems(int status) {
        task(status);
        service.delete(7L, ACTOR);
        verify(items).delete(any(Wrapper.class));
        verify(tasks).deleteById(7L);
        verify(guard).requireTerminal(GenerationKind.IMAGE, 7L);
    }

    @Test
    void rejectsTerminalTaskWhileWorkerIsStillWriting() {
        task(3);
        doThrow(new BusinessException(409, "远端仍在结束处理中")).when(guard).requireTerminal(GenerationKind.IMAGE, 7L);
        assertThatThrownBy(() -> service.delete(7L, ACTOR))
                .isInstanceOf(BusinessException.class).hasMessageContaining("结束处理中");
        verifyNoInteractions(items);
    }

    @Test
    void refusesToContinueAWorkerWhenItsStatusWriteLostToDeletion() {
        when(tasks.selectById(7L)).thenReturn(ImageTask.builder().id(7L).status(3).build());
        when(tasks.updateById(any(ImageTask.class))).thenReturn(0);
        assertThatThrownBy(() -> service.updateStatus(7L, 1, null))
                .isInstanceOf(BusinessException.class).hasMessageContaining("已删除");
    }

    private void task(int status) {
        when(tasks.selectForUpdate(7L)).thenReturn(ImageTask.builder()
                .id(7L).userId(2L).modelId(5L).taskId("task-7").status(status).build());
    }
}
