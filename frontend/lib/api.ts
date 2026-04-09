import type { EventDetailResponse, EventsResponse } from "@/types/api";

const API_BASE_URL =
  process.env.API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      Accept: "application/json",
    },
    next: {
      revalidate: 60,
    },
  });

  if (!response.ok) {
    throw new Error(`API request failed for ${path}: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getEvents(): Promise<EventsResponse> {
  return request<EventsResponse>("/v1/events");
}

export function getEventDetail(eventId: string): Promise<EventDetailResponse> {
  return request<EventDetailResponse>(`/v1/events/${encodeURIComponent(eventId)}`);
}

export function formatEventMeta(date: string | null, venue: string | null, location: string | null) {
  return [date, venue, location].filter(Boolean).join(" · ");
}

export function formatTimestamp(timestamp: string | null) {
  if (!timestamp) {
    return "Odds refresh unavailable";
  }

  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return "Odds refresh unavailable";
  }

  return parsed.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function formatPercent(value: number | null, digits = 0) {
  if (value === null || Number.isNaN(value)) {
    return "N/A";
  }
  return `${(value * 100).toFixed(digits)}%`;
}
