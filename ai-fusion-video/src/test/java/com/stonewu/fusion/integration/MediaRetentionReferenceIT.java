package com.stonewu.fusion.integration;

import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/** 优化后的存在性检查必须保留权威视图的所有非文档引用及其失效条件。 */
@Testcontainers
class MediaRetentionReferenceIT extends MediaResourceTestContext {
    @Test void directReferencesAgreeWithTheViewIncludingTicketsAndProducerIdentity() throws Exception {
        String source = media();
        jdbc.update("INSERT INTO afv_media_source(resource_id,source_type,source_id,source_version,source_field) VALUES(?,'ASSET','71',1,'image')", source);
        retained(source, "BUSINESS_SOURCE");

        String binding = media();
        jdbc.update("INSERT INTO afv_media_binding(resource_type,resource_id,field_name,media_id,resource_version,bound_by) VALUES('ASSET',71,'image',?,1,71)", binding);
        retained(binding, "BUSINESS_BINDING");

        String preset = media();
        jdbc.update("INSERT INTO afv_media_preset_binding(preset_key,media_id,published_by) VALUES('retention-test',?,71)", preset);
        retained(preset, "PRESET");

        String previewSource = media(), previewOutput = media();
        String job = mediaJob();
        jdbc.update("INSERT INTO afv_media_preview_job(job_id,source_resource_id,source_version,profile,sample_at_ms,output_resource_id) VALUES(?,?,1,'image-thumb-v1',0,?)", job, previewSource, previewOutput);
        retained(previewSource, "PREVIEW_SOURCE");
        retained(previewOutput, "PREVIEW_OUTPUT");

        String ticket = media();
        jdbc.update("""
                INSERT INTO afv_media_access_ticket(token_hash,resource_id,resource_version,actor_user_id,purpose,expires_at)
                VALUES(?,?,1,71,'BROWSER',TIMESTAMPADD(HOUR,1,UTC_TIMESTAMP(6)))
                """, "b".repeat(64), ticket);
        retained(ticket, "ACCESS_TICKET");
        jdbc.update("UPDATE afv_media_access_ticket SET revoked=TRUE WHERE resource_id=?", ticket);
        unreferenced(ticket);
        jdbc.update("UPDATE afv_media_access_ticket SET revoked=FALSE,expires_at=TIMESTAMPADD(SECOND,-1,UTC_TIMESTAMP(6)) WHERE resource_id=?", ticket);
        unreferenced(ticket);

        String submission = generationSubmission();
        String generationInput = media();
        jdbc.update("""
                INSERT INTO afv_remote_generation_job(id,submission_id,item_id,item_index,model_id,actor_user_id,
                    state,phase,input_json,next_poll_at,deadline_at,created_at,updated_at)
                VALUES(?,?,1,0,71,71,'SUCCEEDED','COMPLETE',?,UTC_TIMESTAMP(6),TIMESTAMPADD(HOUR,1,UTC_TIMESTAMP(6)),UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))
                """, id(), submission, mapper.writeValueAsString(Map.of("inputMediaVersions", Map.of(generationInput, 1))));
        retained(generationInput, "GENERATION_INPUT");

        String mediaOutput = media();
        jdbc.update("UPDATE afv_media_resource SET request_id=? WHERE id=?", "job:" + job + ":result", mediaOutput);
        retained(mediaOutput, "MEDIA_JOB_OUTPUT");
        jdbc.update("UPDATE afv_media_resource SET created_by=72 WHERE id=?", mediaOutput);
        unreferenced(mediaOutput);

        String generationOutput = media();
        jdbc.update("UPDATE afv_media_resource SET request_id='generation:image:90001:result' WHERE id=?", generationOutput);
        retained(generationOutput, "GENERATION_OUTPUT");
        jdbc.update("UPDATE afv_media_resource SET created_by=72 WHERE id=?", generationOutput);
        unreferenced(generationOutput);
    }

    private String media() throws Exception {
        return upload(id(), png, "引用验收.png", null, USER).id();
    }

    private String mediaJob() {
        String job = id();
        jdbc.update("""
                INSERT INTO afv_media_job(id,kind,project_id,target_id,target_version,actor_user_id,request_id,request_hash,
                    context_json,input_json,checkpoint_json,deadline_at)
                VALUES(?,'IMAGE_PREVIEW',71,71,1,71,?,?,'{}','{}','{}',TIMESTAMPADD(HOUR,1,UTC_TIMESTAMP(6)))
                """, job, id(), "a".repeat(64));
        return job;
    }

    private String generationSubmission() {
        String submission = id();
        jdbc.update("""
                INSERT INTO afv_generation_submission(id,request_id,scope_key,request_hash,kind,task_id,public_task_id,
                    actor_user_id,context_json,model_snapshot_json,api_config_id,api_config_hash,state,created_at,updated_at)
                VALUES(?,?,?,?,'IMAGE',90001,?,71,'{}','{}',71,?,'SUCCEEDED',UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))
                """, submission, id(), "a".repeat(64), "a".repeat(64), id(), "a".repeat(64));
        return submission;
    }

    private void retained(String mediaId, String expectedKind) {
        assertThat(repository.retentionReference(mediaId)).contains(expectedKind);
        assertThat(jdbc.queryForList("SELECT anchor_kind FROM afv_media_reference WHERE media_id=?", String.class, mediaId))
                .contains(expectedKind);
    }

    private void unreferenced(String mediaId) {
        assertThat(repository.retentionReference(mediaId)).isEmpty();
        assertThat(jdbc.queryForList("SELECT anchor_kind FROM afv_media_reference WHERE media_id=?", String.class, mediaId)).isEmpty();
    }
}
