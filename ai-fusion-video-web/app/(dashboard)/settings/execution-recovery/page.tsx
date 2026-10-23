import type { Metadata } from "next";
import { ExecutionRecoveryWorkspace } from "./_components/recovery-workspace";

export const metadata: Metadata = { title: "执行核实" };
export default function ExecutionRecoveryPage() { return <ExecutionRecoveryWorkspace />; }
