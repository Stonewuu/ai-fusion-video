"use client";

import { useState } from "react";
import { Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { mediaId } from "@/lib/api/media";
import { MediaDeleteDialog } from "./media-delete-dialog";

export function MediaReferenceActions({ source, label, onRemove }: {
  source: string;
  label: string;
  onRemove: () => void;
}) {
  const id = mediaId(source);
  const [deleting, setDeleting] = useState(false);
  return (
    <>
      <div className="absolute inset-x-1 top-1 flex justify-end gap-2">
        {id && (
          <Button variant="secondary" size="icon-xs" title={"删除" + label + "素材"} aria-label={"删除" + label + "素材"}
            onClick={() => setDeleting(true)}><Trash2 /></Button>
        )}
        <Button variant="secondary" size="icon-xs" title={"移除" + label} aria-label={"移除" + label} onClick={onRemove}><X /></Button>
      </div>
      {id && deleting && (
        <MediaDeleteDialog media={{ id, name: label }} onClose={() => setDeleting(false)} onDeleted={(result) => {
          if (result.mode === "PERMANENT") {
            onRemove();
          }
        }} />
      )}
    </>
  );
}
