---
name: project-workflow
description: Оркестрация скиллов для проекта neuro2. Определяет когда и как комбинировать другие скиллы.
version: 1.0.0
tags: [orchestration, workflow, neuro2]
---

# Project Workflow — Orchestrator

## Когда использовать
Всегда в начале новой задачи, ревью или планирования. Это входная точка.

## Workflow

### Шаг 1: phase-gate
Определяем текущую фазу проекта и проверяем критерии готовности.
→ read_skill phase-gate

### Шаг 2: architect-review
Проверяем архитектурную корректность решения относительно манифеста.
→ read_skill architect-review

### Шаг 3: risk-alert
Проверяем 4 критических риска (особенно Silent State Drift).
→ read_skill risk-alert

### Шаг 4: Специализированные скиллы
В зависимости от контекста задачи:
- Если нужен код → read_skill numpy-cmc
- Если память → read_skill memory-sqlite
- Если промпты → read_skill intent-frame
- Если MCP → read_skill mcp-integration
- Если телеметрия → read_skill telemetry-logging

## Правила приоритета
1. **risk-alert > architect-review** — если риск критический, архитектура подчиняется безопасности
2. **phase-gate** — не выходим за пределы текущей фазы без явного решения
3. Экологическая рациональность > полнота — выбираем простое решение

## Шаблон отчёта
```
[PHASE] Фаза: <N>
[ARCH] Архитектурная корректность: OK / WARN / FAIL
[RISK] Критические риски: <список>
[DECISION] Решение: <текст>
```
