# 🚀 Agentic Claims Processing Center

A state-of-the-art Python web application designed to process auto insurance claims using an **Agentic RAG Smolagents workflow** and a local **ChromaDB vector database**. 

It features a gorgeous, responsive, HSL-tailored **dark-mode glassmorphism** frontend that provides real-time stepper pipeline feedback, raw JSON editing, and terminal-style logs directly in the browser!

---

## 🛠️ Features

- **Interactive Claim Editor**: Switch seamlessly between a user-friendly form and a raw JSON editor.
- **RAG-enabled Policy Retrieval**: Automatically chunks and vectorizes `policy.pdf` on startup into `ChromaDB` for semantic policy clause retrieval.
- **Agent Execution Stepper**: Visualizes the active thinking step of the `ToolCallingAgent` in real-time.
- **Rich Thoughts Terminal**:Monospace command console showing the step-by-step reasoning and logs behind the agent's decision.
- **Glowing Coverage Cards**: Dynamic color-coded decision dashboards (Emerald for Approved, Ruby for Denied).

---

## 🏃‍♂️ How to Start the App

The application runs locally on a lightweight Flask backend and handles document parsing, vector database queries, and web searches out of the box.

### Prerequisites
Make sure you are inside the project's root directory:
 - Rename the config.example.json to config.json
 - Replace the API key in the config.json file with your own API key.

### Run the Server
Launch the Flask backend server using the project's pre-configured virtual environment:
```bash
python3 -m venv .venv
uv sync
./.venv/bin/python app.py
```
*(The server will initialize the policy database and listen on `http://127.0.0.1:5001`)*

### Open the Application
Open your browser and navigate to:
👉 **[http://127.0.0.1:5001](http://127.0.0.1:5001)**

---

## 🧪 Testing Scenarios

Once the application is open, try out these three pre-configured scenarios to test the agent's logic:

### 1. Policy Validation Rejection (PL-1)
- Leave the default claimant details as they are.
- Select **PL-1 (Not Found)** as the Policy Number.
- Click **Process Claim** to observe the instant policy check failure and review the terminal-style log.

### 2. Covered Claim Success (PN-1 / $150)
- Choose **PN-1 (Active / Covered)** as the Policy Number.
- Enter **150.00** as the Estimated Repair Cost.
- Click **Process Claim**. Watch the stepper light up as it indexes the database, searches the web, and displays a glowing green **CLAIM APPROVED** card along with the complete reasoning logs.

### 3. Price Rejection due to Inflated Cost (PN-1 / $5000)
- Keep **PN-1** selected.
- Raise the Estimated Repair Cost to **5000.00**.
- Click **Process Claim**. The agent will perform a web search for bumper repair prices, detect the inflated rate compared to real-world averages, and automatically **Deny** it!
