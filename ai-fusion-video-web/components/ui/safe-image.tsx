import React, { useState } from "react";
import AssetTypePlaceholder from "@/components/dashboard/asset-type-placeholder";

export interface SafeImageProps extends Omit<React.ImgHTMLAttributes<HTMLImageElement>, "src"> {
  src?: string | null;
  fallbackType?: "avatar" | "scene" | "prop" | "image";
  fallback?: React.ReactNode;
}

export function SafeImage({
  src,
  fallbackType = "image",
  fallback,
  className,
  alt,
  ...props
}: SafeImageProps) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const isError = Boolean(src && failedSrc === src);

  // 映射 fallbackType 到 AssetTypePlaceholder 的 type
  const mappedType = (() => {
    switch (fallbackType) {
      case "avatar":
        return "character";
      case "scene":
        return "scene";
      case "prop":
        return "prop";
      case "image":
      default:
        return "image";
    }
  })();

  if (isError) {
    if (fallback) return <>{fallback}</>;
    return (
      <AssetTypePlaceholder
        type={mappedType}
        className={className}
        isError={true}
        onClick={props.onClick}
      />
    );
  }

  if (!src) {
    if (fallback) return <>{fallback}</>;
    return (
      <AssetTypePlaceholder
        type={mappedType}
        className={className}
        onClick={props.onClick}
      />
    );
  }

  return (
    <img
      {...props}
      src={src}
      className={className}
      alt={alt}
      onError={(event) => {
        setFailedSrc(src);
        props.onError?.(event);
      }}
      onLoad={(event) => {
        setFailedSrc(null);
        props.onLoad?.(event);
      }}
    />
  );
}
