"use client";
import { useEffect, useRef, useState } from "react";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";
import { Button } from "@/components/ui/button";
import { useMediaUrl } from "@/hooks/use-media-url";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { getMediaResource } from "@/lib/api/media";

export function CanvasDocumentPreview({
  source,
  pdf,
}: {
  source: string;
  pdf: boolean;
}) {
  const access = useMediaUrl(source);
  if (access.error)
    return (
      <p role="alert" className="p-4 text-sm text-destructive">
        {access.error}
      </p>
    );
  if (!access.url)
    return (
      <p role="status" className="p-4 text-sm text-muted-foreground">
        正在读取文件…
      </p>
    );
  return pdf ? (
    <PdfPages url={access.url} />
  ) : (
    <TextFile source={source} url={access.url} />
  );
}

function TextFile({ source, url }: { source: string; url: string }) {
  const [result, setResult] = useState<{
    url: string;
    text?: string;
    error?: string;
  } | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      const metadata = await getMediaResource(source);
      if (metadata.content.kind !== "TEXT") throw new Error("此文件不是文本");
      if (metadata.content.byteSize > 10 * 1024 * 1024)
        throw new Error("文本超过 10 MiB，请下载后阅读或拆分文件");
      const response = await fetch(url, {
        signal: controller.signal,
        cache: "no-store",
        referrerPolicy: "no-referrer",
      });
      if (!response.ok) throw new Error(`文件读取失败（${response.status}）`);
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength !== metadata.content.byteSize)
        throw new Error("文本内容大小与登记不一致");
      const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
      if (!controller.signal.aborted) setResult({ url, text });
    })().catch((failure) => {
      if (!controller.signal.aborted)
        setResult({ url, error: getApiErrorMessage(failure) });
    });
    return () => controller.abort();
  }, [source, url]);
  if (result?.url !== url)
    return (
      <p role="status" className="p-4 text-sm text-muted-foreground">
        正在读取文本…
      </p>
    );
  return result.error ? (
    <p role="alert" className="p-4 text-sm text-destructive">
      {result.error}
    </p>
  ) : (
    <pre className="whitespace-pre-wrap break-words p-4 font-sans text-sm leading-relaxed">
      {result.text}
    </pre>
  );
}

function PdfPages({ url }: { url: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [loaded, setLoaded] = useState<{
    url: string;
    document?: PDFDocumentProxy;
    error?: string;
  } | null>(null);
  const [page, setPage] = useState(1);
  const [rendered, setRendered] = useState<{
    key: string;
    error?: string;
    text?: string;
  } | null>(null);
  const document = loaded?.url === url ? loaded.document : undefined;
  const currentPage = document ? Math.min(page, document.numPages) : page;
  const key = `${url}:${currentPage}`;
  useEffect(() => {
    let cancelled = false;
    let dispose: (() => void) | undefined;
    void (async () => {
      const pdfjs = await import("pdfjs-dist");
      if (cancelled) return;
      pdfjs.GlobalWorkerOptions.workerSrc = `/pdf-assets/build/pdf.worker.min.mjs?v=${pdfjs.version}`;
      const base = "/pdf-assets/";
      const loading = pdfjs.getDocument({
        url,
        enableXfa: false,
        cMapUrl: `${base}cmaps/`,
        cMapPacked: true,
        standardFontDataUrl: `${base}standard_fonts/`,
        wasmUrl: `${base}wasm/`,
        disableAutoFetch: true,
        disableStream: true,
        maxImageSize: 16000000,
      });
      dispose = () => {
        void loading.destroy();
      };
      const document = await loading.promise;
      if (!cancelled) setLoaded({ url, document });
    })().catch((failure) => {
      if (!cancelled) setLoaded({ url, error: getApiErrorMessage(failure) });
    });
    return () => {
      cancelled = true;
      dispose?.();
    };
  }, [url]);
  useEffect(() => {
    if (!document || !canvas.current) return;
    let cancelled = false;
    let task: RenderTask | undefined;
    const target = canvas.current;
    void (async () => {
      const pdfPage = await document.getPage(currentPage);
      if (cancelled) return;
      const original = pdfPage.getViewport({ scale: 1 });
      const scale = Math.min(2, 1600 / original.width, 4096 / original.height);
      const viewport = pdfPage.getViewport({ scale });
      target.width = Math.ceil(viewport.width);
      target.height = Math.ceil(viewport.height);
      task = pdfPage.render({ canvas: target, viewport });
      await task.promise;
      const content = await pdfPage.getTextContent();
      if (!cancelled)
        setRendered({
          key,
          text: content.items
            .map((item) => ("str" in item ? item.str : ""))
            .join(" "),
        });
    })().catch((failure) => {
      if (!cancelled) setRendered({ key, error: getApiErrorMessage(failure) });
    });
    return () => {
      cancelled = true;
      task?.cancel();
    };
  }, [document, currentPage, key]);
  const error = loaded?.url === url ? loaded.error : null;
  return (
    <div className="space-y-3 p-4">
      <div className="flex items-center justify-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={!document || currentPage <= 1}
          onClick={() => setPage(currentPage - 1)}
        >
          上一页
        </Button>
        <span className="text-sm">
          {currentPage} / {document?.numPages ?? "…"}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={!document || currentPage >= document.numPages}
          onClick={() => setPage(currentPage + 1)}
        >
          下一页
        </Button>
      </div>
      {(error || (rendered?.key === key && rendered.error)) && (
        <p role="alert" className="text-sm text-destructive">
          {error ?? rendered?.error}
        </p>
      )}
      {!error && rendered?.key !== key && (
        <p role="status" className="text-sm text-muted-foreground">
          正在渲染页面…
        </p>
      )}
      <canvas
        ref={canvas}
        hidden={!!error || rendered?.key !== key || !!rendered.error}
        className="mx-auto h-auto max-w-full"
        aria-label={`文档第 ${currentPage} 页`}
      />
      {rendered?.key === key && rendered.text && (
        <details>
          <summary className="cursor-pointer text-sm">页面文字</summary>
          <p className="whitespace-pre-wrap text-sm leading-relaxed">
            {rendered.text}
          </p>
        </details>
      )}
    </div>
  );
}
