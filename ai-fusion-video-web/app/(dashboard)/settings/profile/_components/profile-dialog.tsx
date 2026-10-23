"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import * as authApi from "@/lib/api/auth";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useConfirm } from "@/components/ui/confirm-dialog";

export function ProfileDialog({ onClose, user, onSaved }: {
  onClose: () => void;
  user: { nickname: string; email: string; phone: string };
  onSaved: () => Promise<void>;
}) {
  const { confirm } = useConfirm();
  const [original] = useState(user);
  const [form, setForm] = useState(user);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dirty = JSON.stringify(form) !== JSON.stringify(original);
  const close = async () => {
    if (saving) return;
    if (dirty && !await confirm({ title: "放弃未保存的资料修改？", confirmText: "放弃修改", cancelText: "继续编辑", variant: "warning" })) return;
    onClose();
  };
  const save = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); setSaving(true); setError(null);
    try {
      await authApi.updateProfile({ nickname: form.nickname.trim(), email: form.email.trim(), phone: form.phone.trim() });
      await onSaved(); toast.success("个人资料已保存"); onClose();
    } catch (failure) { setError(failure instanceof Error ? failure.message : "保存失败"); }
    finally { setSaving(false); }
  };
  return <Dialog open onOpenChange={(open) => { if (!open) void close(); }}>
    <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-md">
      <DialogHeader className="shrink-0"><DialogTitle>编辑个人信息</DialogTitle><DialogDescription>修改你的昵称、邮箱和手机号</DialogDescription></DialogHeader>
      <form onSubmit={save} className="flex min-h-0 flex-col gap-6">
        <div className="flex min-h-0 flex-col gap-4 overflow-y-auto px-1 py-1">
          {([
            ["nickname", "昵称", "text", "请输入昵称"], ["email", "邮箱地址", "email", "请输入邮箱"], ["phone", "手机号码", "tel", "请输入手机号"],
          ] as const).map(([field, label, type, placeholder]) => <div key={field} className="flex flex-col gap-2">
            <Label htmlFor={`profile-${field}`}>{label}</Label>
            <Input id={`profile-${field}`} type={type} placeholder={placeholder} value={form[field]} disabled={saving} onChange={(event) => setForm({ ...form, [field]: event.target.value })} />
          </div>)}
          {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter className="shrink-0"><Button type="button" variant="outline" disabled={saving} onClick={() => void close()}>取消</Button><Button type="submit" disabled={saving || !dirty}>{saving && <Loader2 className="animate-spin" />}保存</Button></DialogFooter>
      </form>
    </DialogContent>
  </Dialog>;
}
