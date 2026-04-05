<div align="center">

# ALIA — AI Avatar

### Adaptive Language Intelligence Avatar

*Developed fot  **VITAL Lab** · Esprit School of Engineering · 4DS1 · 2025–2026*


---

## Overview

**ALIA**  is an intelligent AI avatar system built for **VITAL Lab**. It combines Large Language Models (LLMs), Computer Vision (CNN), and Natural Language Processing (NLP) within a modular multi-agent architecture.

The system is designed to analyze multimodal user input — text and visual data — coordinate specialized AI agents, and generate structured, context-aware responses in real time.

---

## Features

| Feature | Description |
|---|---|
| LLM Reasoning | Natural language dialogue and decision-making via large language models |
| Visual Analysis | CNN-based perception including emotion and expression detection |
| NLP Pipeline | Intent recognition, entity extraction, and semantic understanding |
| Multi-Agent System | Modular agents for parallel task handling and coordination |
| Structured Output | Responses formatted as JSON for downstream integration |
| Extensible Design | Plug-and-play architecture for adding new agents and capabilities |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    ALIA Avatar UI                   │
│            (Real-time Interaction Layer)            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           Multi-Agent Orchestration Core            │
│     Task routing · Memory · Agent coordination      │
└───┬────────────────┬──────────────┬─────────────────┘
    │                │              │
┌───▼───┐     ┌──────▼──────┐  ┌───▼──────────┐
│  LLM  │     │     NLP     │  │  CNN Vision  │
│ Agent │     │  Pipeline   │  │    Module    │
│       │     │             │  │              │
│       │     │             │  │  OpenCV +    │
│       │     │             │  │  DeepFace    │
└───────┘     └─────────────┘  └──────────────┘
```

**LLM Agent** — Core reasoning engine. Handles open-domain conversation, task planning, and response generation using an external LLM API.

**NLP Pipeline** — Processes raw text through tokenization, named entity recognition (NER), intent classification, and sentiment analysis.

**CNN Vision Module** — Analyzes visual inputs (images, video frames) for emotion detection, facial expression recognition, and scene understanding via OpenCV and DeepFace.

**Orchestration Core** — Routes tasks to the appropriate agents, manages context/memory, and assembles final structured JSON output.

---

## Technologies

| Layer | Tools |
|---|---|
| Language | Python 3.10+, Java |
| LLM | LLM API |
| Computer Vision | OpenCV, DeepFace |
| NLP | NLP libraries |
| Output Format | JSON |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip` package manager
- An LLM API key (set as environment variable)

### 1. Clone the repository

```bash
git clone https://github.com/mouhameddhia/Esprit-PI-4DS1-2025-2026-ALIA-AI-Avatar.git
cd Esprit-PI-4DS1-2025-2026-ALIA-AI-Avatar
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the root of the project:

```env
LLM_API_KEY=your_api_key_here
```

### 5. Run the application

```bash
python src/main.py
```

---

## Project Structure

```
ALIA-AI-Avatar/
├── src/
│   ├── main.py              # Application entry point
│   ├── agents/              # Multi-agent system modules
│   ├── nlp/                 # NLP pipeline components
│   ├── vision/              # CNN & computer vision modules
│   └── llm/                 # LLM interface and prompt management
├── data/                    # Sample inputs and test assets
├── outputs/                 # Generated JSON responses
├── requirements.txt
├── .env.example
└── README.md
```

---

## Use Cases

- **Healthcare assistance** — Emotion-aware patient interaction support
- **Emotion & stress detection** — Real-time affective state analysis
- **Decision support tools** — Context-aware recommendations
- **Human-computer interaction** — Research in conversational AI systems

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please follow clean code practices and document any new modules or agents you add.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Notes

> This project is under **active development**. Some modules may be incomplete or subject to change. Contributions and feedback are encouraged.

---

<div align="center">

*Built for  VITAL Lab · Esprit School of Engineering*

</div>
