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

interface GeneratedPackageSummary {
  name?: string;
  entrypoint?: string | null;
  instructions?: string | null;
  files?: PackageFileSummary[];
  metadata?: Record<string, unknown>;
}

interface PackageArtifactEntry {
  agent: string;
  subtask_id: string;
  summary?: string | null;
  created_at?: string;
  artifact: {
    storage: string;
    archive_base64?: string;
    bucket?: string;
    object?: string;
    size_bytes?: number;
    package?: GeneratedPackageSummary;
  };
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

const TERMINAL_STATUSES: TaskStatus[] = ["completed", "failed", "cancelled"];
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

const statusLabel: Record<TaskStatus, string> = {
  pending: "Ожидает",
  in_progress: "В работе",
  completed: "Завершено",
  failed: "Ошибка",
  cancelled: "Отменено"
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

function App() {
  const [query, setQuery] = useState("");
  const [taskId, setTaskId] = useState<string>("");
  const [manualTaskId, setManualTaskId] = useState("");
  const [task, setTask] = useState<TaskStatusResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeActionSubtaskId, setActiveActionSubtaskId] = useState<string | null>(null);

  const baseUrl = useMemo(() => API_BASE_URL || window.location.origin, []);

  const fetchTaskStatus = useCallback(
    async (id: string) => {
      try {
        const response = await fetch(`${baseUrl}/tasks/${id}`);
        if (!response.ok) {
          throw new Error(`Ошибка загрузки статуса: ${response.status}`);
        }
        const data: TaskStatusResponse = await response.json();
        const normalized: TaskStatusResponse = {
          ...data,
          subtasks: data.subtasks ?? [],
          errors: data.errors ?? [],
          artifacts: data.artifacts ?? {},
          events: data.events ?? []
        };
        setTask(normalized);
        if (TERMINAL_STATUSES.includes(data.status)) {
          setIsPolling(false);
        }
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : "Не удалось получить статус задачи");
        setIsPolling(false);
      }
    },
    [baseUrl]
  );

  const postAction = useCallback(
    async (path: string, payload?: unknown) => {
      const response = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: payload ? JSON.stringify(payload) : undefined
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
        setError("Load or create a task before managing subtasks.");
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
      const reasonInput = window.prompt("Optional reason for retry?", "");
      if (reasonInput === null) {
        return;
      }
      const reason = reasonInput.trim() ? reasonInput.trim() : undefined;
      void runSubtaskAction(subtask.subtask_id, () =>
        postAction(
          `/tasks/${taskId}/subtasks/${subtask.subtask_id}/retry`,
          reason ? { reason } : undefined
        )
      );
    },
    [postAction, runSubtaskAction, taskId]
  );

  const handleCancelSubtask = useCallback(
    (subtask: SubtaskResponse) => {
      const reasonInput = window.prompt("Optional reason for cancel?", "");
      if (reasonInput === null) {
        return;
      }
      const reason = reasonInput.trim() ? reasonInput.trim() : undefined;
      void runSubtaskAction(subtask.subtask_id, () =>
        postAction(
          `/tasks/${taskId}/subtasks/${subtask.subtask_id}/cancel`,
          reason ? { reason } : undefined
        )
      );
    },
    [postAction, runSubtaskAction, taskId]
  );

  const handlePrioritizeSubtask = useCallback(
    (subtask: SubtaskResponse) => {
      const current = subtask.priority ?? 0;
      const priorityInput = window.prompt("Set new priority (0-100)", String(current));
      if (priorityInput === null) {
        return;
      }
      const parsed = Number(priorityInput);
      if (!Number.isFinite(parsed) || parsed < 0 || parsed > 100) {
        setError("Priority must be between 0 and 100.");
        return;
      }
      const reasonInput = window.prompt("Optional reason for priority change", "");
      if (reasonInput === null) {
        return;
      }
      const payload: Record<string, unknown> = {
        priority: Math.round(parsed)
      };
      if (reasonInput.trim()) {
        payload.reason = reasonInput.trim();
      }
      void runSubtaskAction(subtask.subtask_id, () =>
        postAction(
          `/tasks/${taskId}/subtasks/${subtask.subtask_id}/prioritize`,
          payload
        )
      );
    },
    [postAction, runSubtaskAction, taskId]
  );

  const handleDownloadPackage = useCallback(
    (entry: PackageArtifactEntry, index: number) => {
      const artifact = entry.artifact;
      if (artifact.storage !== "inline_base64" || !artifact.archive_base64) {
        setError("Download is available only for inline artifacts.");
        return;
      }
      try {
        const byteString = atob(artifact.archive_base64);
        const bytes = new Uint8Array(byteString.length);
        for (let i = 0; i < byteString.length; i += 1) {
          bytes[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: "application/zip" });
        const baseName = entry.artifact.package?.name?.trim() || `package-${index + 1}`;
        const safeBaseName = baseName.replace(/[^a-zA-Z0-9._-]+/g, "_");
        const filename = safeBaseName.toLowerCase().endsWith(".zip") ? safeBaseName : `${safeBaseName}.zip`;
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
      } catch (err) {
        console.error(err);
        setError("Failed to download package archive.");
      }
    },
    [setError]
  );

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

  const handleSubmitTask = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("Введите запрос");
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      const response = await fetch(`${baseUrl}/tasks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: query.trim() })
      });

      if (!response.ok) {
        throw new Error(`Ошибка создания задачи: ${response.status}`);
      }

      const data: TaskCreationResponse = await response.json();
      setTaskId(data.task_id);
      setManualTaskId(data.task_id);
      setTask(null);
      setIsPolling(true);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Не удалось создать задачу");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLoadTask = async () => {
    if (!manualTaskId.trim()) {
      setError("Введите идентификатор задачи");
      return;
    }
    setError(null);
    setTask(null);
    setTaskId(manualTaskId.trim());
    setIsPolling(true);
  };

  return (
    <div className="layout">
      <header className="app-header">
        <div>
          <h1>Multi-Agent Task Viewer</h1>
          <p>Отправляйте запросы в мультиагентную систему и следите за прогрессом в едином интерфейсе.</p>
        </div>
        {taskId && (
          <div className="header-meta">
            <span className="meta-label">Текущая задача:</span>
            <span className="meta-value">{taskId}</span>
          </div>
        )}
      </header>

      <main className="content">
        <section className="panel">
          <h2>Создать новую задачу</h2>
          <form className="form" onSubmit={handleSubmitTask}>
            <label htmlFor="query">Запрос</label>
            <textarea
              id="query"
              placeholder="Например: Исследовать современные методы суммаризации видео..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              rows={4}
            />
            <div className="form-actions">
              <button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Отправка..." : "Запустить задачу"}
              </button>
            </div>
          </form>
        </section>

        <section className="panel">
          <h2>Загрузить задачу по ID</h2>
          <div className="manual-load">
            <input
              type="text"
              placeholder="UUID задачи"
              value={manualTaskId}
              onChange={(event) => setManualTaskId(event.target.value)}
            />
            <button type="button" onClick={handleLoadTask}>
              Загрузить
            </button>
          </div>
          <p className="help-text">
            Можно вставить идентификатор из ответа мультиагента. Интерфейс начнет опрашивать статус автоматически.
          </p>
        </section>

        {error && (
          <section className="panel error-panel">
            <h2>Ошибка</h2>
            <p>{error}</p>
          </section>
        )}

        {task && (
          <>
            <section className="panel">
              <div className="status-header">
                <div>
                  <h2>Статус задачи</h2>
                  <p className="status-text">{statusLabel[task.status]}</p>
                </div>
                <div className="progress">
                  <span>Прогресс</span>
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
                  <span className="meta-label">Идентификатор задачи</span>
                  <span className="meta-value">{task.task_id}</span>
                </div>
              </div>
            </section>

            <section className="panel">
              <h2>Подзадачи</h2>
              {task.subtasks.length === 0 ? (
                <p>Подзадачи пока не определены.</p>
              ) : (
                <div className="subtask-grid">
                  {task.subtasks.map((subtask) => {

                    const isBusy = activeActionSubtaskId === subtask.subtask_id;

                    const canRetry = subtask.status !== "in_progress";

                    const canCancel = subtask.status === "in_progress" || subtask.status === "pending";

                    const retryCount = subtask.retry_count ?? 0;

                    const priority = subtask.priority ?? 0;

                    return (

                      <article key={subtask.subtask_id} className={clsx("subtask-card", subtask.status)}>

                        <header>

                          <span className="agent">{subtask.agent_type}</span>

                          <span className={clsx("status-badge", subtask.status)}>{statusLabel[subtask.status]}</span>

                        </header>

                        {subtask.manual && (

                          <span className="manual-tag">

                            Manual{subtask.manual_triggered_at ? ` @ ${formatDate(subtask.manual_triggered_at)}` : ""}

                          </span>

                        )}

                        <p className="description">{subtask.description}</p>

                        <div className="subtask-meta-row">

                          <span>Priority: {priority}</span>

                          <span>Retries: {retryCount}</span>

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

                          <p className="error-text">Last error: {subtask.last_error}</p>

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

                          <button

                            type="button"

                            onClick={() => handlePrioritizeSubtask(subtask)}

                            disabled={isBusy}

                          >

                            Prioritize

                          </button>

                        </div>

                      </article>

                    );

                  })}
                </div>
              )}
            </section>

            {packages.length > 0 && (

              <section className="panel">

                <h2>Generated Packages</h2>

                <div className="package-grid">

                  {packages.map((pkg, index) => (

                    <article key={`${pkg.subtask_id}-${index}`} className="package-card">

                      <header>

                        <span className="package-name">{pkg.artifact.package?.name ?? `Package ${index + 1}`}</span>

                        <span className="agent">{pkg.agent}</span>

                      </header>

                      <p className="package-summary">{pkg.summary ?? "No summary available."}</p>

                      <ul className="package-details">

                        <li>Files: {pkg.artifact.package?.files?.length ?? 0}</li>

                        {pkg.artifact.package?.entrypoint && (

                          <li>

                            Entrypoint: <code>{pkg.artifact.package.entrypoint}</code>

                          </li>

                        )}

                        {pkg.artifact.package?.instructions && (

                          <li>Run: {pkg.artifact.package.instructions}</li>

                        )}

                        {typeof pkg.artifact.size_bytes === "number" && (

                          <li>Size: {(pkg.artifact.size_bytes / 1024).toFixed(1)} KB</li>

                        )}

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

                            Stored in GCS ({pkg.artifact.bucket}/{pkg.artifact.object})

                          </span>

                        )}

                      </div>

                    </article>

                  ))}

                </div>

              </section>

            )}



<section className="panel">
              <h2>Сводный ответ</h2>
              {task.result ? <MarkdownViewer content={task.result} /> : <p>Результат пока не готов.</p>}
            </section>
             {events.length > 0 && (

              <section className="panel">

                <h2>Activity Log</h2>

                <ul className="events-list">

                  {events.slice().reverse().map((evt, index) => {

                    const rawTimestamp = evt["timestamp"];

                    const timestamp = typeof rawTimestamp === "string" ? rawTimestamp : undefined;

                    const rawAction = evt["action"] ?? evt["status"];

                    const action = String(rawAction ?? "event");

                    const rawMessage = evt["error"] ?? evt["message"];

                    const message = rawMessage ? String(rawMessage) : "";

                    const rawSubtaskId = evt["subtask_id"];

                    const subtaskId = typeof rawSubtaskId === "string" ? rawSubtaskId : undefined;

                    return (

                      <li key={`${subtaskId ?? "event"}-${index}`}>

                        <span className="event-time">{formatDate(timestamp)}</span>

                        <span className="event-action">{action}</span>

                        {subtaskId && <span className="event-subtask">{subtaskId}</span>}

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
            <h2>Ошибки</h2>
            <ul>
              {task.errors.map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </section>
        )}
      </main>

      <footer className="footer">
        <span>База API: {baseUrl}</span>
        {isPolling && <span className="polling-indicator">Обновление...</span>}
      </footer>
    </div>
  );
}

export default App;
