ALTER TABLE `afv_ai_model`
  ADD COLUMN `multimodal_input_types` json NULL COMMENT '支持的多模态输入类型：image/video/audio/file' AFTER `support_vision`,
  ADD COLUMN `multimodal_input_transports` json NULL COMMENT '各输入类型支持的传输方式：url/base64' AFTER `multimodal_input_types`;

UPDATE `afv_ai_model`
SET `multimodal_input_types` = CASE
      WHEN `support_vision` = b'1' THEN JSON_ARRAY('image')
      ELSE JSON_ARRAY()
    END,
    `multimodal_input_transports` = CASE
      WHEN `support_vision` = b'1' THEN JSON_OBJECT('image', JSON_ARRAY('url', 'base64'))
      ELSE JSON_OBJECT()
    END;

ALTER TABLE `afv_ai_model`
  MODIFY COLUMN `multimodal_input_types` json NOT NULL COMMENT '支持的多模态输入类型：image/video/audio/file',
  MODIFY COLUMN `multimodal_input_transports` json NOT NULL COMMENT '各输入类型支持的传输方式：url/base64';
