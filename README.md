## LLM-Powered CLI Automation Agent

A Python-based AI agent that leverages LLM function-calling to plan, execute, and iterate on multi-step tasks using dynamic tool invocation.

This project implements an agentic workflow system where a large language model can:

- Analyze user requests
- Select appropriate tools
- Execute actions
- Evaluate results
- Continue iterating until task completion

The agent operates through a controlled loop, enabling multi-step reasoning and tool chaining.

### Features

#### LLM Function Calling

- Uses structured tool schemas to enable safe and predictable tool execution

#### Multi-Step Agent Loop

- Iteratively processes model outputs and feeds results back for continued reasoning

#### Tooling System

- File inspection (list directories, read files)
- File writing and modification
- Python script execution

#### Prompt Engineering

- System prompts guide task decomposition and tool selection

#### Structured JSON Pipelines

- Standardized communication between LLM outputs and backend functions

#### Error Handling & Validation

- Handles invalid function calls and runtime failures gracefully

#### Secure Execution Environment

- Restricts file operations to a controlled working directory

#### Token Monitoring

- Tracks API usage for debugging and performance optimization


### Architecture
1. User Input (CLI)
2. LLM (Gemini API)
3. Function Call Decision
4. Tool Execution Layer (Python)
5. Structured Response (JSON)
6. Loop Back into LLM

### Tech Stack

- Python
- Google Gemini API
- Function Calling (Tool Schemas)
- REST APIs
- JSON Data Pipelines

### Example Workflow

#### User Prompt:

> "Find all Python files and run the main script"

### Agent Execution:

1. Lists files in directory
2. Identifies relevant Python files
3. Executes selected script
4. Returns output

### Key Concepts Demonstrated

- Agentic AI workflows
- Tool-based reasoning
- Prompt engineering strategies
- API-driven automation
- Structured LLM integration
- Safe execution environments

#### Future Improvements

- Add retry logic for failed tool calls
- Implement logging and observability
- Support external API integrations
- Introduce parallel tool execution
- Add memory for long-running workflows


This project demonstrates how LLMs can move beyond text generation into actionable systems that automate real workflows, bridging the gap between AI models and production-ready software.