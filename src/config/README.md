# Config

Единый источник настраиваемых параметров хоста (CONSTITUTION §2.2):
`dt` (10 Гц, эмоц. контур), человеческие константы аффекта
(`valence_tau≈1 с`, `stress_leak≈0.01/с`), `gamma_max`, `time_scale`,
пороги дрейфа, колонки.

S2: `MemoryConfig` — эмбеддер (`embedder_mode` auto/fake/api), БД памяти,
порог записи эпизода, `prior_dim`.

S3: `SpeechConfig` — речь/LLM (register, f_threshold, history_turns, ...).
S4: `HomeostasisConfig` — сетепоинты battery/resources/cpu, порог рефлекса,
множители throttle; `PolicyConfig` — `Preferences`, режим хоста, attention_gate.
S5: `SocialConfig` — ToM (пороги узнавания/конфликта/паузы).
S6: `AutonomyConfig` — самоконтроль/консолидация/драйв/факторы.
S7: `presets.py` — именованные детерминированные пресеты
(`baseline`/`stress`/`dialogue`/`autonomy`/`long-horizon`/`cooperative`) +
`load_preset(name, override=TOML)`; `configs/*.toml` — примеры override.

`HostConfig` собирает `EnergyConfig` + `ColumnParams` + `AttractorConfig` +
`MemoryConfig` + `SpeechConfig` + `HomeostasisConfig` + `PolicyConfig` +
`SocialConfig` + `AutonomyConfig` и прокидывается в `build_host_loop`.
Значения — стартовые (калибровка S1–S4).

См. `SPEC.md`, `PLAN.md` и ADR-0006 (временные шкалы).