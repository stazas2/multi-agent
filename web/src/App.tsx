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
  started_at?: string;
  completed_at?: string;
}

interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  progress: number;
  subtasks: SubtaskResponse[];
  result?: string | null;
  errors: string[];
  trace_id: string;
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

  const baseUrl = useMemo(() => API_BASE_URL || window.location.origin, []);

  const fetchTaskStatus = useCallback(
    async (id: string) => {
      try {
        const response = await fetch(`${baseUrl}/tasks/${id}`);
        if (!response.ok) {
          throw new Error(`Ошибка загрузки статуса: ${response.status}`);
        }
        const data: TaskStatusResponse = await response.json();
        setTask(data);
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

  useEffect(() => {
    if (!taskId || !isPolling) {
      return;
    }

    let isActive = true;

    const poll = async () => {
      if (!isActive) return;
      await fetchTaskStatus(taskId);
    };

    // первая загрузка без ожидания
    poll();

    const interval = setInterval(poll, 3500);

    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [taskId, isPolling, fetchTaskStatus]);

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
                  {task.subtasks.map((subtask) => (
                    <article key={subtask.subtask_id} className={clsx("subtask-card", subtask.status)}>
                      <header>
                        <span className="agent">{subtask.agent_type}</span>
                        <span className={clsx("status-badge", subtask.status)}>{statusLabel[subtask.status]}</span>
                      </header>
                      <p className="description">{subtask.description}</p>
                      <dl>
                        <div>
                          <dt>Начало</dt>
                          <dd>{formatDate(subtask.started_at)}</dd>
                        </div>
                        <div>
                          <dt>Завершение</dt>
                          <dd>{formatDate(subtask.completed_at)}</dd>
                        </div>
                      </dl>
                      {subtask.error && <p className="error-text">Ошибка: {subtask.error}</p>}
                      {subtask.result && (
                        <details>
                          <summary>Результат агента</summary>
                          <pre>{JSON.stringify(subtask.result, null, 2)}</pre>
                        </details>
                      )}
                    </article>
                  ))}
                </div>
              )}
            </section>

            <section className="panel">
              <h2>Сводный ответ</h2>
              {task.result ? <MarkdownViewer content={task.result} /> : <p>Результат пока не готов.</p>}
            </section>
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
