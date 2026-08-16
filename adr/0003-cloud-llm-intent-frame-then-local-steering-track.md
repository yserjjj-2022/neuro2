# ADR-0003: Трек речи — Cloud LLM (Intent-Frame) → Local SLM (Activation Steering)

## Date
2025-01-16

## Status
Accepted

## Context
Хосту нужно генерировать ответы в диалоге. Два трека развития речи:
1. **Intent-Frame**: структурированный промпт → облачный LLM API
2. **Activation Steering**: прямой injection состояния в residual stream локальной SLM

Вопрос: какой трек использовать на Фазе 1?

## Decision
**Фаза 1**: Cloud LLM + Intent-Frame (быстрый старт, валидация концепции)
**Фаза 2+**: Переход на локальную SLM + Activation Steering

### Почему Intent-Frame на Фазе 1
- **Готовность**: API уже есть (OpenAI, Anthropic и т.д.), не нужно обучать модель
- **Качество**: облачные LLM дают лучший baseline для валидации архитектуры хоста
- **Intent-Frame как артефакт**: позволяет отладить структуру промпта (цель, аффект, прецеденты, стиль) до перехода на steering
- **Скорость**: не нужно ждать загрузки локальной модели (Gemma-3-4B / Qwen3-1.7B на M2 8GB)

### Почему переход на Steering в Фазе 2
- **Мегапромпты хрупки**: модели "сжимают, переинтерпретируют, пропускают" инструкции при накоплении
- **Steering bypasses text**: injection в residual stream не подвержен compression
- **EvolvingSteeringMemory**: накопление вектора характера из правок оператора требует прямого доступа к model internals
- **Приватность**: локальная модель — данные не покидают устройство

## Consequences
### Положительные
- Быстрый MVP: Фазу 1 можно пройти за 1-2 недели
- Intent-Frame как промежуточный артефакт помогает понять, какие именно инструкции "теряются"
- Плавный переход: когда steering готов, Intent-Frame заменяется на hooks

### Отрицательные
- Зависимость от внешнего API на Фазе 1 (стоимость, latency, privacy)
- Intent-Frame требует тщательного тестирования: нужно убедиться, что LLM не игнорирует часть инструкций
- Steering требует access to forward hooks — не все модели это поддерживают одинаково

### Tradeoffs
- **Качество vs контроль**: Intent-Frame даёт качество, но теряет контроль над инструкциями; Steering даёт контроль, но требует зрелой инфраструктуры
- **Стоимость vs приватность**: API стоит денег, но данные уходят наружу; локальная модель бесплатна и приватна, но требует hardware

## Confidence
High. Манифест прямо указывает на эту эволюцию (раздел 4). Empirical evidence: мегапромпты действительно деградируют при длине >2000 токенов.

## References
- [Манифест, раздел 4 — Speech Track](../host_architecture_manifest.md)
- [Activation Steering survey](https://arxiv.org/abs/2310.07556)
- [EvolvingSteeringMemory concept](../host_architecture_manifest.md)
