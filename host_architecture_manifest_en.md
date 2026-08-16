# Adaptive Host Architecture Manifest

## 1. Core Concept

The system is built on principles analogous to the neocortex: canonical columnar microcircuits, continuous time, predictive coding, and free energy minimization (Active Inference). The goal is not a monolithic LLM, but a modular, temporally consistent host-interlocutor/co-player capable of situational learning, endowed with a sense of time, emotional dynamics, stable character, and the ability for autonomous evolution.

The key principle is **ecological rationality**: the system does not compute everything in a strictly Bayesian manner, but computes only what is necessary, conserving computational resources (event-triggered inference, sparse factorization, heuristic fast paths).

---

## 2. Architectural Scheme

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    1. SENSORY AFFORDANCES & MCP LAYER                  │
 │  • Incoming text / actions   • MCP Resources (weather, location,       │
 │  • Circadian time & pauses     system status, battery)                │
 │  • Device somatics (CPU)     • MCP Tools (epistemic probing of the     │
 │                                world: search, news)                     │
 └────────────────────────────────────────────────────────────────────────┘
                                      │ Perception vector u(t)
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │       2. EPISTEMIC VIGILANCE BARRIER                                   │
 │  • Protection against cognitive drift: verification of sarcasm,        │
 │    manipulation, contradictory statements                              │
 │  • Dynamic precision assignment to channel (Precision weighting γ)     │
 └────────────────────────────────────────────────────────────────────────┘
                                      │ Weighted input
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │          3. CORTICAL COLUMN ENSEMBLE (PARALLEL CMC FABRIC)             │
 │                                                                        │
 │  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
 │  │ Column: Tone    │   │ Column: Rhythm  │   │ Columns: Meanings,   │  │
 │  │ & Emotions      │   │ & Dialogue Pauses│  │ ToM, MCP Telemetry   │  │
 │  │ (L4→L5/6→L2/3)  │   │                 │   │                      │  │
 │  └────────┬────────┘   └────────┬────────┘   └──────────┬───────────┘  │
 │           └─────────────────────┼───────────────────────┘              │
 │                                 ▼                                      │
 │                  Lateral Voting (L2/3, k-WTA)                          │
 │        Free energy F(t) calculation, valence (-dF/dt),              │
 │        allostatic stress and confidence (precision)                  │
 │                                                                        │
 │  Discrete Upper Layer (macro-modes, rare calls):                     │
 │  Sparse factorization [Host Mode | Partner Assessment | Task]        │
 └───────────────────┬────────────────────────────────┬───────────────────┘
                      │                                │
                      ▼                                ▼
 ┌────────────────────────────────────┐   ┌───────────────────────────────┐
 │    4. MEMORY & CONSOLIDATION (CLS)  │   │ 5. ECOLOGICAL SPEECH ACTUATOR │
 │  • Episodic buffer (SQLite +       │   │  • Event-Triggered LLM call   │
 │    sqlite-vec)                     │   │  • Intent-Frame generation /  │
 │  • Night sleep: noise pruning,     │   │    Steering-vector generation │
 │    Structure Learning (Schemas)    │   │  • Joint/autonomous action    │
 │  • EvolvingSteeringMemory —        │   │    via MCP                    │
 │    accumulation of "character"     │   │                               │
 │    vector                          │   │                               │
 └────────────────────────────────────┘   └───────────────────────────────┘
```

---

## 3. Key Subsystems

### A. Cortical Tissue (Canonical Microcircuits)
- Node: layers L4 (input) → L5/6 (state generator x(t)) → L2/3 (prediction error e(t)).
- Column specialization by narrow reality projections: tone, rhythm, tasks, ToM, MCP sensors.
- Voting via lateral inhibition (k-WTA) — consensus without a central controller.
- Parallelism: batch of columns in a single tensor [N_columns, In_Dim, State_Dim], computed in one SIMD tick on CPU.
- Scaling is horizontal and linear (O(N)) — new columns do not increase complexity of existing ones (unlike quadratic transformer growth).

### B. Continuous Time and Interoception
- Circadian harmonics (sin/cos phases of day), interval pause counter, subjective tempo (accelerates with rising F(t)).
- Valence = -dF/dt; allostatic stress = integral of F(t) over time; precision (gamma) = inverse variance of input signal.
- These four signals form an affective modifier passed to the speech actuator.

### C. Goal Setting (Hierarchical Active Inference)
- Root Priors — constant values (coherence, integrity, verification).
- Task Goal — temporary dynamic task attractors with TTL and priority; at idle, task energy grows, host reminds about it.
- Epistemic Drive — in absence of tasks, host seeks areas of maximum uncertainty and explores them autonomously (curiosity).

### D. Social Circuit
- Simplified Theory of Mind: scalar/vector partner_trust, partner_state — assessment of consent, ambiguity, conflict.
- Joint Agency — joint target attractors for game/work scenarios (host covers, doesn't just execute).
- Vigilance Gate — new operator statements are marked as hypotheses until confirmed by practice; protection against epistemic drift.

### E. Memory (Complementary Learning Systems)
- Fast buffer: SQLite + sqlite-vec, embedding-precedents of edits and situations (sub-millisecond per query).
- Consolidation during sleep: active pruning of irrelevant episodes + Structure Learning (generalization of repeating patterns into schemas).
- Why SQLite, not Postgres/pgvector: embedded architecture gives 0.1-0.5 ms latency and zero background process — optimal for a single local user; migration to Postgres is justified only for multi-user/distributed scenarios.

### F. Ecological Rationality and Hybrid Computation
- Event-Triggered Active Inference: full recalculation is triggered only when F(t) exceeds threshold; otherwise system drifts by inertia.
- Recognition Heuristic: typical situations (greeting, confirmation) are handled by instant template without LLM call.
- Sparse factorization instead of a single POMDP matrix: 3 independent small factors (mode/partner/task) instead of combinatorial explosion in pymdp.
- pymdp used sparingly — for 4-8 discrete macro-states; all continuous dynamics on NumPy/SciPy.

### G. MCP as Senses and Actions
- MCP Resources — passive background sensory input (weather, location, device status), mixed into u(t) on slow tick.
- MCP Tools — active epistemic actions: invoked when uncertainty is better reduced by external query than by inference.
- MCP Actions — executive tools, physical/digital impact on the environment.
- Available MCP list forms the host affordance map — the space of what it can do in principle.

---

## 4. Speech Track: From Mega-Prompt to Grown Speech

```
 B1: Mega-Prompt (Intent-Frame) -> B2: Activation Steering -> B3: Steering + CMLA -> B4: Emergent/Reactive Speech
```

- **B1. Mega-Prompt bridge.** The core forms a structured Intent-Frame (utterance goal, affect, precedents, style) and passes it to a cloud LLM API. Risk: model may compress, re-interpret, skip part of instructions as they accumulate.
- **B2. Activation Steering on local SLM.** Transition to a local model (Gemma-3-4B / Qwen3-1.7B, MLX/HF transformers on M2 8GB) gives direct access to forward hooks: core state vector x(t) is injected into the model residual stream, bypassing text. EvolvingSteeringMemory accumulates and refines the character vector from operator contrastive correction pairs — so the host speech style is literally grown from experience, not rewritten prompts. Limitation: controls only behavioral axes (tone, confidence, conciseness), not facts.
- **B3. Hybrid Steering + CMLA.** In parallel, a corpus of speech directed at the host is assembled — a narrow, situational analogue of Child-Directed Speech from real interaction. External/local LLM acts as mentor: generates reference answers, on which the host compact production model learns to map meaning to speech, using computational modelling of language acquisition methodology. This is an explicit teacher-learner closed loop, where the language model itself is the source of training signals for the younger system.
- **B4. Reactive Columnar Speech (horizon).** Experimental Emergent Communication direction: host columns develop their own compositional protocol for solving tasks without pre-assigned grammar (Neural Iterated Learning, phoneme-lexical learning from scratch). Honest assessment: still academic frontier without production tools; result — not necessarily natural language, but an efficient internal protocol requiring a final translation layer into external speech.

---

## 5. Roadmap (4 phases) with Speech Track Synchronization

| Phase | Core | Speech Track | Readiness Criteria |
| :--- | :--- | :--- | :--- |
| **1. Reactive MVP** (1-2 wks) | Continuous core x(t) on NumPy, F(t) calc, episodic memory in SQLite, ecological Embeddings/LLM API call | B1: Mega-Prompt (Intent-Frame) | Host remembers corrections instantly, changes tone at divergence |
| **2. Sensory Agent** (2-4 wks) | Circadian time, interval pauses, MCP integration (resources+tools), dynamic task attractors | B2 (start): transition to local SLM, hooks prep | Host remembers time of day, reminds about unfinished tasks |
| **3. Co-Player (ToM)** (1-2 mos) | Column ensemble (CMC Fabric) with lateral voting, partner model, Vigilance Gate, Joint Agency | B2 (mature) -> B3 (start): EvolvingSteeringMemory works, CDS corpus collection | Host covers player, recognizes manipulation and bluff |
| **4. Mature Autonomous Subject** (2-3 mos) | Night sleep (active pruning + Structure Learning), epistemic drive, role priority removal (gamma_persona -> 0) | B3 (mature) -> B4 (experimental): production model learns from LLM mentor, first Emergent Communication experiments | Host works autonomously for weeks, optimizes memory, develops own style |

---

## 6. Critical Assessment (Risks and Limitations)

1. **Free Energy Principle — not a ready recipe, but a controversial hypothesis.** Critics point to low falsifiability of FEP; retrospective projects show active inference often doesn't yield measurable advantage over standard RL/heuristics. Recommendation: keep a simple empirical alternative at each stage and compare actual gains.
2. **Scaling generative models remains an open problem** — hybrid discrete/continuous core bypasses but doesn't solve this complexity; empirical calibration of thresholds and coefficients will be required.
3. **Mega-prompts are fragile** — models compress, re-interpret, skip instructions as they accumulate; this is a direct reason to move to Activation Steering as early as possible.
4. **Risk of Silent State Drift** — accumulative memory and evolving priors can silently drift host behavior from norms. Explicit telemetry logging ([F | Valence | State]) and a meta-cognitive self-diagnostic layer ("I'm confused, manual reset needed") is necessary, which in the base manifest version was implicit and must become an architected first-class component.
5. **Steering doesn't replace knowledge** — controls tone and style, not factual accuracy; for facts, MCP/RAG remain.

---

## 7. Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| Sensory input and actions | MCP servers (Resources + Tools) | Senses and manipulators |
| Input adapter | Embeddings API (or local MiniLM ONNX) | Text to continuous vector |
| Columnar core | NumPy / PyTorch CPU (SIMD batching) | x(t) dynamics, F(t), emotions, ToM |
| Discrete upper-level logic | inferactively-pymdp (sparing, 4-8 states) | Macro-modes, goal setting |
| Episodic memory | SQLite + sqlite-vec | Precedents, consolidation, EvolvingSteeringMemory |
| Speech actuator (Phase 1) | External LLM API + Intent-Frame | Quick start |
| Speech actuator (Phase 2+) | Local SLM (Gemma-3-4B / Qwen3-1.7B, MLX/HF transformers) + Activation Steering | Tone control without text instructions |
| Speech actuator (Phase 3-4) | Production model, learned from LLM mentor (CMLA) | Grown-from-within speech |

---

## 8. Life Cycle Modes (Role Flexibility)

| Mode | Role Priorities (gamma_persona) | Goal Source | Freedom |
| :--- | :--- | :--- | :--- |
| Game Avatar | Fixed (game lore) | Curiosity within character | Stable roleplay |
| Adaptive Co-Player | Partially plastic | Joint tasks + ToM | Adaptation to specific person |
| Free Host | Fully plastic (->0) | Epistemic drive, personal experience | Autonomous personality evolution |
