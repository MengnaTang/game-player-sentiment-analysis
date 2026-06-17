# game-player-sentiment-analysis
This project analyzes Google Play reviews of Clash of Clans and Rise of Kingdoms using LLM embeddings (OpenAI) and the PXI questionnaire. It demonstrates a scalable, data-driven pipeline for competitive intelligence in mobile gaming.

The analysis replicates and extends the methodology from the CHI 2026 paper "Mining Player Experience Trends From Game Reviews Using Large Language Models" (Dutta et al.), applying it to a modern F2P context and introducing custom sentiment dimensions tailored to monetisation, ads, customer support, and developer trust.

# Requirements
- Python 3.12+
- OpenAI API key (for generating embeddings)

# Installation
1. Create a conda environment (recommended):
```python
  conda create -n game_analysis python=3.12.3`
  conda activate game_analysis
```
2.Install dependencies:
`pip install -r requirements.txt`
3.Set up your OpenAI API key:
    - Create a `.env` file in the project root with:
    ```text
    OPENAI_API_KEY=your_key_here
    OPENAI_API_BASE=https://api.openai.com/v1
    ```
