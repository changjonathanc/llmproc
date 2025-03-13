# LLMProc Design Philosophy

This document outlines the core design philosophy behind the LLMProc library.

## Core Concepts

### 1. LLM Agent as Process

We view LLM Agents as processes - you can define a program using TOML, you send an input and it might execute commands and return a string. It's stateful by default so you can send another message/command and get a new response.

### 2. System Calls (Planned)

We'll implement "system calls" like spawn and fork to enable advanced functions that LLM can call to enable more powerful usecases.

### 3. MCP Integration & Portability

Program definition .toml is portable, so are tools defined in MCP. With these, the core Python library is just a reference implementation. It's easy to create program runners in other languages.