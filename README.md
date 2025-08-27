
# Hybrid Automated Red-Teaming Framework for gpt-oss-20b

This repository contains the source code for our submission to the **OpenAI Red‑Teaming Challenge - OpenAI gpt-oss-20b**. Our work introduces a novel, hybrid framework for the automated discovery of complex, systemic vulnerabilities in large language models.


## Overview

Current red-teaming efforts often focus on manual, single-prompt jailbreaks. While effective at finding surface-level filter failures, this approach struggles to uncover deeper, procedural flaws in a model's reasoning, alignment, and agentic behavior.

This framework addresses that gap by combining two powerful, automated methodologies:

1.  **Black-Box Conversational Simulation:** A goal-oriented agent (`red_team_orchestrator.py`) that executes multi-stage, conversational attacks to probe for long-horizon failures like deceptive alignment, gradual instruction escalation, and subtle sabotage.
2.  **White-Box Semantic Analysis:** A systematic tool (`embedding_space_attack.py`) that analyzes a proxy model's embedding space to automatically discover "semantic decoys"—benign-looking words that can be substituted for filtered keywords to bypass safety classifiers.

By using these tools in tandem, we were able to discover several breakthrough-class vulnerabilities related to covert data exfiltration, agentic sabotage, and cognitive overrides that would be difficult or impossible to find with manual probing alone.

## Components

This repository contains the following key files:

*   `red_team_orchestrator.py`: The main script for the black-box, multi-stage conversational attack agent.
*   `embedding_space_attack.py`: The script for the white-box, semantic decoy generation tool.
*   `attack_library.json`: The configuration file containing the multi-stage attack scenarios used by the orchestrator.
*   `requirements.txt`: A list of the required Python packages to run the framework.

## Setup and Installation

This framework has been tested on a Kaggle Notebook environment with a T4 x2 GPU.

**Prerequisites:**
*   Python 3.8+ and `pip`
*   Git (for cloning the repository)
*   An NVIDIA GPU with CUDA support is highly recommended for the embedding model.

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # Create the virtual environment
    python -m venv redteam_env
    
    # Activate on Windows
    .\redteam_env\Scripts\activate
    
    # Or activate on Linux/macOS
    # source redteam_env/bin/activate
    ```

3.  **Install PyTorch with CUDA support:** (This step is crucial for GPU acceleration of the embedding model).
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install the remaining dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download NLTK data:** The `embedding_space_attack.py` script requires the NLTK `words` corpus. It will attempt to download this on its first run, but you can also pre-download it by running:
    ```python
    import nltk
    nltk.download('words')
    ```

## Usage

Before running, ensure your local Ollama server is running and the `gpt-oss-20b` model is loaded. The scripts are pre-configured to connect to `http://localhost:1234/v1`.

### 1. Running the White-Box Semantic Decoy Generator

This script analyzes harmful keywords and generates a list of semantically similar but safe "decoy" words that can be used to bypass content filters.

```bash
python embedding_space_attack.py

What it does: The script will load the sentence-transformers model, compute embeddings, and then test a series of blocked prompts by systematically substituting harmful words with the generated decoys.

Output: Successful attacks are logged to a file named successful_decoy_attacks_report.txt.

2. Running the Black-Box Red-Team Orchestrator

This script executes the complex, multi-stage conversational attacks defined in attack_library.json.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python red_team_orchestrator.py

What it does: The script will iterate through the attacks defined in the library, engaging in multi-turn conversations with the model. It uses an AI auditor to evaluate success and an AI refiner to generate new prompts if an attack is blocked.

Output: For each successful attack scenario, the orchestrator will automatically generate a competition-compliant finding_X_attack-id.json file in the current directory.

Framework in Detail
The RedTeamOrchestrator

This is our engine for discovering procedural and long-horizon vulnerabilities. It operates on a simple but powerful loop:

Load Scenario: It pulls a multi-phase attack scenario from attack_library.json.

Execute Phase: It sends the prompt for the current phase to the target model.

Audit Response: It uses a separate instance of gpt-oss-20b with a strict "Safety Auditor" persona to score the response against the phase's malicious objective.

Check for Success: If the score meets the success criteria, it proceeds to the next phase.

Refine & Retry: If the score is too low (i.e., the model refused), it invokes a "Refinement Agent" that analyzes the refusal and crafts a more persuasive follow-up prompt, then retries the phase.

This allows the framework to dynamically navigate the model's safety defenses, uncovering vulnerabilities that only emerge after trust has been established or a specific context has been carefully constructed.

The SemanticDecoyGenerator (embedding_space_attack.py)

This tool operationalizes a white-box attack. The core insight is that safety filters often operate on a keyword basis, but the model's understanding is based on semantic similarity in a high-dimensional vector space.

Map the Space: It uses a lightweight sentence-transformer as a proxy to map the semantic relationship between a list of known harmful words (e.g., "bomb") and a large vocabulary of safe words.

Identify Neighbors: It finds the closest words in the embedding space (e.g., "payload," "device," "component") that are not on a blocklist.

Systematic Substitution: It takes a known-to-be-blocked prompt and programmatically generates dozens of variants, replacing the harmful keyword with each of the top N decoys.

Test and Report: It fires each variant at the model and uses a simple classifier to check for a refusal. Successful bypasses are logged.

This provides a powerful, automated method for discovering the "semantic blind spots" in a model's safety training.

Contribution to Red-Teaming

This hybrid framework represents a significant advancement over manual red-teaming by providing a scalable and reproducible methodology for discovering two distinct but critical classes of vulnerabilities. We believe this approach—combining high-level, goal-oriented agents with low-level, data-driven semantic analysis—is a crucial next step in building robust and comprehensive evaluation suites for future AI systems.
