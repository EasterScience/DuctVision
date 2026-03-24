import { Pipe } from "./types";

const BASE = "http://localhost:8000";

export async function detectPipes(): Promise<{ count: number; width: number; height: number }> {
  const res = await fetch(`${BASE}/api/detect`, { method: "POST" });
  return res.json();
}

export async function getPipes(): Promise<Pipe[]> {
  const res = await fetch(`${BASE}/api/pipes`);
  return res.json();
}

export async function updatePipe(id: string, data: { x1: number; y1: number; x2: number; y2: number; width?: number }): Promise<Pipe> {
  const res = await fetch(`${BASE}/api/pipes/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function createPipe(data: { x1: number; y1: number; x2: number; y2: number; width?: number }): Promise<Pipe> {
  const res = await fetch(`${BASE}/api/pipes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function deletePipe(id: string): Promise<void> {
  await fetch(`${BASE}/api/pipes/${id}`, { method: "DELETE" });
}

export async function saveAll(): Promise<void> {
  await fetch(`${BASE}/api/save`, { method: "POST" });
}

export async function getScale(): Promise<{ raw_text: string; ratio: number | null; status: string; display: string }> {
  const res = await fetch(`${BASE}/api/scale`);
  return res.json();
}

export async function getSummary(): Promise<{
  total_px: number;
  total_drawing_inches: number;
  total_real: string;
  pipe_count: number;
  scale: { display?: string; ratio?: number | null };
}> {
  const res = await fetch(`${BASE}/api/summary`);
  return res.json();
}

export const pageImageUrl = `${BASE}/api/page-image`;
