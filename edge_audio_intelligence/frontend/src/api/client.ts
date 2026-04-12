import axios from 'axios';
import type { SceneConfig, SceneResponse } from '../types/scene';
import type { PipelineConfig } from '../types/pipeline';

const api = axios.create({
  baseURL: '/api',
});

// ── Scene ──

export async function generateScene(config: SceneConfig): Promise<SceneResponse> {
  const { data } = await api.post<SceneResponse>('/scene/generate', config);
  return data;
}

// ── Pipeline ──

export async function startPipelineRun(config: PipelineConfig): Promise<{ task_id: string }> {
  const { data } = await api.post('/pipeline/run', config);
  return data;
}

export async function getPipelineStatus(taskId: string) {
  const { data } = await api.get(`/pipeline/status/${taskId}`);
  return data;
}

// ── Audio ──

export function getAudioUrl(runId: string, signal: string): string {
  return `/api/audio/${runId}/${signal}`;
}

export async function listAudioSignals(runId: string): Promise<string[]> {
  const { data } = await api.get(`/audio/${runId}`);
  return data.signals;
}

// ── LibriSpeech ──

export interface LibriSpeechUtterance {
  id: string;
  text: string;
  speaker: string;
  path: string;
}

export async function getLibriSpeechIndex(limit = 500): Promise<LibriSpeechUtterance[]> {
  const { data } = await api.get('/librispeech/index', { params: { limit } });
  return data;
}

// ── History ──

export async function getHistory() {
  const { data } = await api.get('/history');
  return data;
}

export async function getRunDetails(runId: string) {
  const { data } = await api.get(`/history/${runId}`);
  return data;
}

export async function deleteRun(runId: string) {
  await api.delete(`/history/${runId}`);
}

// ── WebSocket ──

export function connectPipelineWS(
  taskId: string,
  onMessage: (msg: unknown) => void,
  onClose?: () => void,
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/api/pipeline/ws/${taskId}`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    onMessage(msg);
  };
  ws.onclose = () => onClose?.();
  return ws;
}
