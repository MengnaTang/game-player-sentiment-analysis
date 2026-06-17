# Game Player Sentiment Analysis

A research project that replicates and extends the methodology proposed in **“Mining Player Experience Trends From Game Reviews Using Large Language Models” (Dutta et al., CHI 2026)**.

This project analyzes Google Play reviews of **Clash of Clans** and **Rise of Kingdoms** using LLM embeddings and the PXI (Player Experience Inventory) framework. Building on the original work, it examines how well PXI applies to modern free-to-play mobile games, and explores how player review mining can enrich competitive intelligence and product analysis.


## Research Questions

This project explores three questions:
1. To what extent does the PXI framework capture player concerns in modern F2P mobile games?
2. What additional experience dimensions emerge beyond those covered by PXI?
3. Can large-scale review analysis provide actionable competitive intelligence?


## Methodology

Pipeline:

Google Play Reviews

→ Data Cleaning

→ OpenAI Embedding Generation

→ PXI Similarity Scoring

→ Blind Spot Detection

→ Custom Dimension Discovery

→ Competitive Intelligence Analysis

The implementation follows the CHI 2026 methodology while introducing additional analysis steps designed to identify player concerns that are not represented in the original PXI framework.

### Model Selection

Three player experience questionnaires from the original paper were evaluated:
- PXI
- CORGIS
- AESTHEMOS

PXI showed the highest overall semantic alignment with Google Play reviews and was therefore selected as the primary framework for subsequent analysis.


## Main Contributions

### 1. Replication of CHI 2026 Methodology

Successfully reproduced the PXI-based review analysis pipeline on a new domain:
- Clash of Clans
- Rise of Kingdoms

demonstrating the transferability of the original approach to mobile strategy games.

### 2. Blind Spot Detection

During analysis, a large number of highly negative reviews received low similarity scores across all PXI dimensions.

These reviews revealed player concerns that were not adequately represented by the original PXI framework.

A blind-spot detection procedure was developed to systematically identify such cases.

### 3. Discovery of Additional Experience Dimensions

Through keyword extraction and manual review inspection, four recurring dimensions emerged:
- Pay Progress
- Ad Overload
- Support Quality
- Company Trust

These dimensions appear particularly relevant in free-to-play ecosystems and are not explicitly captured by PXI.

### 4. Competitive Intelligence Application

The project demonstrates how review mining can be used to compare competing games and reveal differences in:
- Monetization perception
- Developer trust
- Customer support reputation
- User satisfaction

providing insights potentially useful for product and business decision-making.

## Key Findings
- PXI does not fully capture monetization-related frustration.
- Blind-spot reviews reveal important player concerns beyond the PXI framework.
- Developer trust emerges as a distinct experience dimension.
- Clash of Clans receives substantially more positive sentiment regarding support quality and developer trust.
- Rise of Kingdoms receives more criticism related to progression and monetization systems.


## Repository Structure
- 1_scrape.ipynb — Scrape Google Play reviews
- 2_cleaned.ipynb — Data cleaning and embedding generation
- 3_analysis.ipynb — PXI analysis, blind-spot detection, and custom dimension discovery


## Requirements
- Python 3.12+
- OpenAI API key

## Installation

Create a conda environment:
```bash
conda create -n game_analysis python=3.12.3
conda activate game_analysis
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file:
```Plain text
OPENAI_API_KEY=your_key_here
OPENAI_API_BASE=https://api.openai.com/v1
```

## Usage

Run notebooks in the following order:
```Plain text
1_scrape.ipynb
↓
2_cleaned.ipynb
↓
3_analysis.ipynb
```

## Outputs

Generated outputs include:
- PXI dimension analysis
- Blind-spot review identification
- Custom sentiment dimensions
- Comparative game analysis
- Figures and summary tables

Results are stored in:
```
results/
```
Precomputed embeddings can be reused from:
```
processed/
```
to avoid repeated API calls.

## Future Work

Potential extensions include:
- Automated blind-spot discovery
- Multi-game benchmarking
- Longitudinal player experience tracking
- User migration analysis across competing games
- Integration with business intelligence dashboards
- Validation against behavioral gameplay data

  
## References

Dutta, S., Oksanen, J., Vakeva, J., Ahmed, S., Kirjonen, M., & Hämäläinen, P. (2026). Mining Player Experience Trends From Game Reviews Using Large Language Models. In *Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26)*. ACM. DOI: [10.1145/3772318.3790760](https://doi.org/10.1145/3772318.3790760)

## License

MIT License

