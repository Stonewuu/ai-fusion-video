package com.stonewu.fusion.service.ai.googleflow;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class GoogleFlowReverseApiSupportTests {

        @Test
        void shouldInferRemoteModelTypeForGoogleFlowAliases() {
                assertEquals(2, GoogleFlowReverseApiSupport.inferRemoteModelType("gemini-3.0-pro-image"));
                assertEquals(2, GoogleFlowReverseApiSupport.inferRemoteModelType("imagen-4.0-generate-preview"));
                assertEquals(3, GoogleFlowReverseApiSupport.inferRemoteModelType("veo_3_1_r2v_fast"));
                assertNull(GoogleFlowReverseApiSupport.inferRemoteModelType("unknown-model"));
        }

    @Test
    void shouldResolveImageAliasToExactModel() {
        AiModel model = AiModel.builder()
                .code("gemini-3.1-flash-image")
                .config("""
                        {
                          "supportedAspectRatios": ["16:9", "9:16", "1:1", "4:3", "3:4"],
                          "supportedSizes": {
                            "1K": {"16:9": "1792x1024"},
                            "2K": {"16:9": "2048x1152"},
                            "4K": {"3:4": "3072x4096"}
                          }
                        }
                        """)
                .build();

        ImageTask task = ImageTask.builder()
                .aspectRatio("3:4")
                .resolution("4K")
                .build();

        GoogleFlowReverseApiSupport.ResolvedImageRequest request = GoogleFlowReverseApiSupport.resolveImageRequest(model, task);

        assertEquals("gemini-3.1-flash-image-three-four-4k", request.actualModelCode());
        assertEquals("3:4", request.aspectRatio());
        assertEquals("4K", request.resolution());
    }

    @Test
    void shouldResolveVideoUpsampleAliasToExactModel() {
        AiModel model = AiModel.builder()
                .code("veo_3_1_t2v_fast")
                .config("""
                        {
                          "supportedResolutions": ["1080P", "4K"],
                          "supportedAspectRatios": ["16:9", "9:16"],
                                                                                                        "minImageInputs": 0,
                                                                                                        "maxImageInputs": 0
                        }
                        """)
                .build();

        VideoTask task = VideoTask.builder()
                .ratio("9:16")
                .resolution("1080P")
                .build();

        GoogleFlowReverseApiSupport.ResolvedVideoRequest request = GoogleFlowReverseApiSupport.resolveVideoRequest(model, task);

        assertEquals("veo_3_1_t2v_fast_portrait_1080p", request.actualModelCode());
        assertEquals("9:16", request.aspectRatio());
        assertEquals("1080P", request.resolution());
        assertEquals(0, request.imageCount());
    }

    @Test
    void shouldResolveI2VUltra4kAliasToExactModel() {
        AiModel model = AiModel.builder()
                .code("veo_3_1_i2v_s_fast_ultra_fl")
                .config("""
                        {
                          "supportedResolutions": ["1080P", "4K"],
                          "supportedAspectRatios": ["16:9", "9:16"],
                                                                                                        "minImageInputs": 1,
                                                                                                        "maxImageInputs": 2
                        }
                        """)
                .build();

        VideoTask task = VideoTask.builder()
                .ratio("16:9")
                .resolution("4K")
                .firstFrameImageUrl("/media/images/a.png")
                .build();

        GoogleFlowReverseApiSupport.ResolvedVideoRequest request = GoogleFlowReverseApiSupport.resolveVideoRequest(model, task);

        assertEquals("veo_3_1_i2v_s_fast_ultra_fl_4k", request.actualModelCode());
        assertEquals(1, request.imageCount());
    }

    @Test
    void shouldExtractMediaUrlsFromFinalContent() {
        assertEquals(
                "https://example.com/image.png",
                GoogleFlowReverseApiSupport.extractImageUrl("![Generated Image](https://example.com/image.png)")
        );
        assertEquals(
                "https://example.com/video.mp4",
                GoogleFlowReverseApiSupport.extractVideoUrl("<video src='https://example.com/video.mp4' controls></video>")
        );
    }
}