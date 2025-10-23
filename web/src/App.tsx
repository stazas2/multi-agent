import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import MarkdownViewer from "./components/MarkdownViewer";
import clsx from "clsx";

type TaskStatus = "pending" | "in_progress" | "completed" | "failed" | "cancelled";

interface SubtaskResponse {
  subtask_id: string;
  parent_task_id: string;
  agent_type: string;
  description: string;
  parameters?: Record<string, unknown>;
  dependencies?: string[];
  status: TaskStatus;
  result?: Record<string, unknown> | null;
  error?: string | null;
  last_error?: string | null;
  started_at?: string;
  completed_at?: string;
  retry_count?: number;
  priority?: number;
  manual?: boolean;
  manual_triggered_at?: string;
}

interface PackageFileSummary {
  path: string;
  executable?: boolean;
}

interface PackageArtifact {
  storage: string;
  archive_base64?: string;
  bucket?: string;
  object?: string;
  size_bytes?: number;
  package?: {
    name?: string;
    files?: PackageFileSummary[];
    entrypoint?: string | null;
    instructions?: string | null;
    metadata?: Record<string, unknown>;
  };
}

interface PackageArtifactEntry {
  agent: string;
  subtask_id: string;
  summary?: string | null;
  created_at?: string;
  artifact: PackageArtifact;
}

interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  progress: number;
  subtasks: SubtaskResponse[];
  result?: string | null;
  errors: string[];
  trace_id: string;
  artifacts?: {
    packages?: PackageArtifactEntry[];
  };
  events?: Array<Record<string, unknown>>;
}

interface TaskCreationResponse {
  task_id: string;
  status: string;
  message: string;
  estimated_time_seconds: number;
  trace_id: string;
}

interface TaskHistoryEntry {
  task_id: string;
  status: TaskStatus;
  progress: number;
  created_at: string;
  updated_at: string;
  has_artifacts: boolean;
  result_preview?: string | null;
  metadata?: Record<string, unknown>;
}

const TERMINAL_STATUSES: TaskStatus[] = ["completed", "failed", "cancelled"];
const ALLOWED_HISTORY_STATUSES: TaskStatus[] = ["pending", "in_progress", "completed", "failed", "cancelled"];
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

const statusLabel: Record<TaskStatus, string> = {
  pending: "Pending",
  in_progress: "In Progress",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

function formatDate(timestamp?: string) {
  if (!timestamp) {
    return "—";
  }
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
}

function formatBytes(size?: number) {
  if (!size || Number.isNaN(size)) {
    return "";
  }
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function App() {
  const [query, setQuery] = useState("");
  const [taskId, setTaskId] = useState<string>("");
  const [manualTaskId, setManualTaskId] = useState("");
  const [task, setTask] = useState<TaskStatusResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeActionSubtaskId, setActiveActionSubtaskId] = useState<string | null>(null);
  const [history, setHistory] = useState<TaskHistoryEntry[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string>("");

  const baseUrl = useMemo(() => API_BASE_URL || window.location.origin, []);

  const fetchHistory = useCallback(async () => {
    try {
      const response = await fetch(`${baseUrl}/tasks?limit=25`);
      if (!response.ok) {
        throw new Error(`Failed to fetch task history: ${response.status}`);
      }
      const payload = await response.json();
      const items = Array.isArray(payload?.items) ? payload.items : [];
      const mapped: TaskHistoryEntry[] = items.map((item: Record<string, unknown>) => {
        const rawStatus = typeof item.status === "string" ? (item.status as TaskStatus) : "pending";
        const status = ALLOWED_HISTORY_STATUSES.includes(rawStatus) ? rawStatus : "pending";
        return {
          task_id: String(item.task_id ?? ""),
          status,
          progress: typeof item.progress === "number" ? item.progress : 0,
          created_at: String(item.created_at ?? ""),
          updated_at: String(item.updated_at ?? ""),
          has_artifacts: Boolean(item.has_artifacts),
          result_preview: typeof item.result_preview === "string" ? item.result_preview : null,
          metadata: (item.metadata as Record<string, unknown> | undefined) ?? {},
        };
      });
      setHistory(mapped);
    } catch (err) {
      console.error(err);
    }
  }, [baseUrl]);

  const fetchTaskStatus = useCallback(
    async (id: string) => {
      try {
        const response = await fetch(`${baseUrl}/tasks/${id}`);
        if (!response.ok) {
        throw new Error(`Failed to fetch task status: ${response.status}`);
        }
        const data: TaskStatusResponse = await response.json();
        const normalized: TaskStatusResponse = {
          ...data,
          subtasks: data.subtasks ?? [],
          errors: data.errors ?? [],
          artifacts: data.artifacts ?? {},
          events: data.events ?? [],
        };
        setTask(normalized);
        if (TERMINAL_STATUSES.includes(data.status)) {
          setIsPolling(false);
          void fetchHistory();
        }
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : "Failed to fetch task status");
        setIsPolling(false);
      }
    },
    [baseUrl, fetchHistory]
  );

  const postAction = useCallback(
    async (path: string, payload?: Record<string, unknown>) => {
      const response = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: payload ? JSON.stringify(payload) : undefined,
      });
      const raw = await response.text();
      if (!response.ok) {
        throw new Error(raw || `Request failed with status ${response.status}`);
      }
      if (!raw) {
        return {};
      }
      try {
        return JSON.parse(raw);
      } catch {
        return {};
      }
    },
    [baseUrl]
  );

  const runSubtaskAction = useCallback(
    async (subtaskId: string, executor: () => Promise<unknown>) => {
      if (!taskId) {
        setError("Please enter a task query.");
        return;
      }
      try {
        setActiveActionSubtaskId(subtaskId);
        await executor();
        await fetchTaskStatus(taskId);
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : "Failed to execute subtask action");
      } finally {
        setActiveActionSubtaskId(null);
      }
    },
    [taskId, fetchTaskStatus]
  );

  const handleRetrySubtask = useCallback(
    (subtask: SubtaskResponse) => {
      const reason = window.prompt("Retry reason (optional):", "");
      if (reason === null) {
        return;
      }
      const payload = reason.trim() ? { reason: reason.trim() } : undefined;
      void runSubtaskAction(subtask.subtask_id, () =>
        postAction(`/tasks/${taskId}/subtasks/${subtask.subtask_id}/retry`, payload)
      );
    },
    [postAction, runSubtaskAction, taskId]
  );

  const handleCancelSubtask = useCallback(
    (subtask: SubtaskResponse) => {
      const reason = window.prompt("Cancel reason (optional):", "");
      if (reason === null) {
        return;
      }
      const payload = reason.trim() ? { reason: reason.trim() } : undefined;
      void runSubtaskAction(subtask.subtask_id, () =>
        postAction(`/tasks/${taskId}/subtasks/${subtask.subtask_id}/cancel`, payload)
      );
    },
    [postAction, runSubtaskAction, taskId]
  );

  const handlePrioritizeSubtask = useCallback(
    (subtask: SubtaskResponse) => {
      const current = subtask.priority ?? 0;
      const input = window.prompt("Set new priority (0-100):", String(current));
      if (input === null) {
        return;
      }
      const parsed = Number(input);
      if (!Number.isFinite(parsed) || parsed < 0 || parsed > 100) {
        setError("Priority must be between 0 and 100.");
        return;
      }
      const reason = window.prompt("Priority change reason (optional):", "");
      const payload: Record<string, unknown> = { priority: Math.round(parsed) };
      if (reason && reason.trim()) {
        payload.reason = reason.trim();
      }
      void runSubtaskAction(subtask.subtask_id, () =>
        postAction(`/tasks/${taskId}/subtasks/${subtask.subtask_id}/prioritize`, payload)
      );
    },
    [postAction, runSubtaskAction, taskId]
  );

  const handleDownloadPackage = useCallback((entry: PackageArtifactEntry, index: number) => {
    const artifact = entry.artifact;
    if (artifact.storage !== "inline_base64" || !artifact.archive_base64) {
      setError("Download is available only for inline (base64) archives.");
      return;
    }
    try {
      const byteString = atob(artifact.archive_base64);
      const bytes = new Uint8Array(byteString.length);
      for (let i = 0; i < byteString.length; i += 1) {
        bytes[i] = byteString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "application/zip" });
      const baseName = artifact.package?.name?.trim() || `package-${index + 1}`;
      const safeName = baseName.replace(/[^0-9a-zA-Z._-]+/g, "_");
      const filename = safeName.toLowerCase().endsWith(".zip") ? safeName : `${safeName}.zip`;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    } catch (err) {
      console.error(err);
      setError("Failed to download archive.");
    }
  }, []);

  const handleSubmitTask = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("Please provide a task ID.");
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      const response = await fetch(`${baseUrl}/tasks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!response.ok) {
        throw new Error(`Failed to submit task: ${response.status}`);
      }

      const data: TaskCreationResponse = await response.json();
      setTaskId(data.task_id);
      setSelectedHistoryId(data.task_id);
      setManualTaskId(data.task_id);
      setTask(null);
      setIsPolling(true);
      await fetchHistory();
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Failed to load task");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLoadTask = async () => {
    if (!manualTaskId.trim()) {
      setError("Enter a task identifier.");
      return;
    }

    const trimmed = manualTaskId.trim();
    setError(null);
    setTask(null);
    setTaskId(trimmed);
    setSelectedHistoryId(trimmed);
    setIsPolling(true);
    await fetchTaskStatus(trimmed);
  };

  const handleSelectHistory = useCallback(
    async (id: string) => {
      setSelectedHistoryId(id);
      setTaskId(id);
      setIsPolling(true);
      await fetchTaskStatus(id);
    },
    [fetchTaskStatus]
  );

  useEffect(() => {
    void fetchHistory();
  }, [fetchHistory]);

  useEffect(() => {
    if (!taskId || !isPolling) {
      return;
    }

    let isActive = true;

    const poll = async () => {
      if (!isActive) return;
      await fetchTaskStatus(taskId);
    };

    poll();

    const interval = setInterval(poll, 3500);

    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [taskId, isPolling, fetchTaskStatus]);

  const packages = task?.artifacts?.packages ?? [];
  const events = task?.events ?? [];

  return (
    <div className="layout">
      <header className="app-header">
        <div>
          <h1>Multi-Agent Task Viewer</h1>
          <p>
            Submit tasks, monitor agent progress, manage subtasks, and download the generated code packages.
          </p>
        </div>
        {taskId && (
          <div className="header-meta">
            <span className="meta-label">Current task_id</span>
            <span className="meta-value">{taskId}</span>
          </div>
        )}
      </header>

      <main className="content">
        <section className="panel">
          <h2>Create a task</h2>
          <form className="form" onSubmit={handleSubmitTask}>
            <label htmlFor="query">Prompt</label>
            <textarea
              id="query"
              placeholder='For example: "Analyse the top 5 AI projects from the last month and suggest improvements"'
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              rows={4}
            />
            <div className="form-actions">
              <button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Submitting..." : "Create"}
              </button>
            </div>
          </form>
        </section>

        <section className="panel">
          <h2>Load an existing task</h2>
          <div className="manual-load">
            <input
              type="text"
              placeholder="Task UUID"
              value={manualTaskId}
              onChange={(event) => setManualTaskId(event.target.value)}
            />
            <button type="button" onClick={handleLoadTask}>
              Load
            </button>
          </div>
          <p className="help-text">
            Paste an existing identifier or pick a task from the history below. Once loaded, you can inspect progress and
            operate subtasks.
          </p>
        </section>

        {history.length > 0 && (
          <section className="panel">
            <details className="history-container">
              <summary>Recent tasks</summary>
              <div className="history-list">
                {history.map((entry) => (
                  <button
                    type="button"
                    key={entry.task_id}
                    className={clsx("history-item", { active: entry.task_id === selectedHistoryId })}
                    onClick={() => handleSelectHistory(entry.task_id)}
                  >
                    <span className="history-item__title">{entry.task_id}</span>
                    <span className={clsx("status-badge", entry.status, "history-item__status")}>
                      {statusLabel[entry.status]}
                    </span>
                    <span className="history-item__date">{formatDate(entry.updated_at)}</span>
                    {entry.has_artifacts && <span className="history-item__tag">ZIP</span>}
                    {entry.result_preview && <span className="history-item__preview">{entry.result_preview}</span>}
                  </button>
                ))}
              </div>
            </details>
          </section>
        )}

        {error && (
          <section className="panel error-panel">
            <h2>Error</h2>
            <p>{error}</p>
          </section>
        )}

        {task && (
          <>
            <section className="panel">
              <div className="status-header">
                <div>
                  <h2>Task status</h2>
                  <p className="status-text">{statusLabel[task.status]}</p>
                </div>
                <div className="progress">
                  <span>Progress</span>
                  <div className="progress-bar">
                    <div className="progress-value" style={{ width: `${task.progress}%` }} />
                  </div>
                  <span className="progress-value-text">{task.progress.toFixed(1)}%</span>
                </div>
              </div>
              <div className="meta-grid">
                <div>
                  <span className="meta-label">Trace ID</span>
                  <span className="meta-value">{task.trace_id}</span>
                </div>
                <div>
                  <span className="meta-label">Task ID</span>
                  <span className="meta-value">{task.task_id}</span>
                </div>
              </div>
            </section>

            <section className="panel">
              <h2>Subtasks</h2>
              {task.subtasks.length === 0 ? (
                <p>No subtasks yet.</p>
              ) : (
                <div className="subtask-grid">
                  {task.subtasks.map((subtask) => {
                    const isBusy = activeActionSubtaskId === subtask.subtask_id;
                    const canRetry = subtask.status !== "in_progress";
                    const canCancel = subtask.status === "pending" || subtask.status === "in_progress";
                    return (
                      <article key={subtask.subtask_id} className={clsx("subtask-card", subtask.status)}>
                        <header>
                          <span className="agent">{subtask.agent_type}</span>
                          <span className={clsx("status-badge", subtask.status)}>{statusLabel[subtask.status]}</span>
                        </header>
                        {subtask.manual && (
                          <span className="manual-tag">
                            MANUAL{subtask.manual_triggered_at ? ` @ ${formatDate(subtask.manual_triggered_at)}` : ""}
                          </span>
                        )}
                        <p className="description">{subtask.description}</p>

                        <div className="subtask-meta-row">
                          <span>Priority: {subtask.priority ?? 0}</span>
                          <span>Retries: {subtask.retry_count ?? 0}</span>
                        </div>

                        <dl>
                          <div>
                            <dt>Started</dt>
                            <dd>{formatDate(subtask.started_at)}</dd>
                          </div>
                          <div>
                            <dt>Finished</dt>
                            <dd>{formatDate(subtask.completed_at)}</dd>
                          </div>
                        </dl>

                        {subtask.last_error && subtask.last_error !== subtask.error && (
                          <p className="error-text">Previous error: {subtask.last_error}</p>
                        )}
                        {subtask.error && <p className="error-text">Error: {subtask.error}</p>}

                        {subtask.result && (
                          <details>
                            <summary>View result payload</summary>
                            <pre>{JSON.stringify(subtask.result, null, 2)}</pre>
                          </details>
                        )}

                        <div className="subtask-actions">
                          <button
                            type="button"
                            onClick={() => handleRetrySubtask(subtask)}
                            disabled={isBusy || !canRetry}
                          >
                            Retry
                          </button>
                          <button
                            type="button"
                            onClick={() => handleCancelSubtask(subtask)}
                            disabled={isBusy || !canCancel}
                          >
                            Cancel
                          </button>
                          <button type="button" onClick={() => handlePrioritizeSubtask(subtask)} disabled={isBusy}>
                            Priority
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </section>

            <section className="panel">
              <h2>Final answer</h2>
              {task.result ? <MarkdownViewer content={task.result} /> : <p>The final response is not ready yet.</p>}
            </section>

            {packages.length > 0 && (
              <section className="panel">
                <h2>Generated packages</h2>
                <div className="package-grid">
                  {packages.map((pkg, index) => {
                    const pkgInfo = pkg.artifact.package;
                    const fileCount = pkgInfo?.files?.length ?? 0;
                    const sizeHint = formatBytes(pkg.artifact.size_bytes);
                    return (
                      <article key={`${pkg.subtask_id}-${index}`} className="package-card">
                        <header>
                          <span className="package-name">{pkgInfo?.name ?? `Package ${index + 1}`}</span>
                          <span className="agent">{pkg.agent}</span>
                        </header>
                        {pkg.summary && <p className="package-summary">{pkg.summary}</p>}
                        <ul className="package-details">
                          <li>Files: {fileCount}</li>
                          {pkgInfo?.entrypoint && (
                            <li>
                              Entrypoint: <code>{pkgInfo.entrypoint}</code>
                            </li>
                          )}
                          {pkgInfo?.instructions && <li>How to run: {pkgInfo.instructions}</li>}
                          {sizeHint && <li>Size: {sizeHint}</li>}
                        </ul>
                        <div className="package-actions">
                          <button
                            type="button"
                            onClick={() => handleDownloadPackage(pkg, index)}
                            disabled={pkg.artifact.storage !== "inline_base64"}
                          >
                            Download ZIP
                          </button>
                          {pkg.artifact.storage === "gcs" && (
                            <span className="package-note">
                              Stored in GCS: {pkg.artifact.bucket}/{pkg.artifact.object}
                            </span>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              </section>
            )}

            {events.length > 0 && (
              <section className="panel">
                <h2>Activity log</h2>
                <ul className="events-list">
                  {events
                    .slice()
                    .reverse()
                    .slice(0, 25)
                    .map((evt, index) => {
                      const timestamp = typeof evt.timestamp === "string" ? evt.timestamp : undefined;
                      const action = String(evt.action ?? evt.status ?? "event");
                      const subtask = typeof evt.subtask_id === "string" ? evt.subtask_id : "";
                      const message =
                        typeof evt.error === "string"
                          ? evt.error
                          : typeof evt.message === "string"
                          ? evt.message
                          : "";
                      return (
                        <li key={`${subtask || "event"}-${index}`}>
                          <span className="event-time">{formatDate(timestamp)}</span>
                          <span className="event-action">{action}</span>
                          <span className="event-subtask">{subtask}</span>
                          <span className="event-message">{message}</span>
                        </li>
                      );
                    })}
                </ul>
              </section>
            )}
          </>
        )}

        {task && task.errors.length > 0 && (
          <section className="panel error-panel">
            <h2>Errors</h2>
            <ul>
              {task.errors.map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </section>
        )}
      </main>

      <footer className="footer">
        <span>API: {baseUrl}</span>
        {isPolling && <span className="polling-indicator">Updating...</span>}
      </footer>
    </div>
  );
}
export default App;

