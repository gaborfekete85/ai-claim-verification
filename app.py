import json
import os
import time
import re
import csv
import datetime
import io
import contextlib
from typing import List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from pydantic import BaseModel, Field

from smolagents import (
    tool,
    ToolCallingAgent,
    WebSearchTool,
    OpenAIServerModel,
    PromptTemplates,
    PlanningPromptTemplate,
    ManagedAgentPromptTemplate,
    FinalAnswerPromptTemplate,
)

from flask import Flask, render_template, request, jsonify

# 1. Load configuration and setup environment
file_name = 'config.json'
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Configuration file {file_name} not found.")

with open(file_name, 'r') as file:
    config = json.load(file)
    os.environ['OPENAI_API_KEY'] = config.get("API_KEY")
    os.environ["OPENAI_BASE_URL"] = config.get("OPENAI_API_BASE")

# 2. Setup LLM and Embeddings
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base=os.environ['OPENAI_BASE_URL'],
    api_key=os.environ['OPENAI_API_KEY'],
)

sentence_transformer_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 3. Initialize ChromaDB and Index Policy Document
chroma_client = chromadb.Client()
collection_name = "auto_insurance_policy"

try:
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=sentence_transformer_ef)
except Exception as e:
    collection = chroma_client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef)

policy_file_path = "policy.pdf"
if not os.path.exists(policy_file_path):
    raise FileNotFoundError(f"Policy PDF file {policy_file_path} not found.")

if collection.count() == 0:
    print("[INIT]: Indexing policy.pdf into ChromaDB...")
    loader = PyPDFLoader(policy_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    policy_chunks = text_splitter.split_documents(docs)
    chunks = [chunk.page_content for chunk in policy_chunks]
    metadata = [chunk.metadata for chunk in policy_chunks]
    chunk_ids = [f"chunk_{i}" for i in range(len(policy_chunks))]
    collection.add(
        documents=chunks,
        metadatas=metadata,
        ids=chunk_ids,
    )
    print(f"[INIT]: Indexed {collection.count()} chunks successfully.")
else:
    print(f"[INIT]: Using existing vector collection with {collection.count()} chunks.")

# 4. Pydantic Schemas
class ClaimInfo(BaseModel):
    """Extracted insurance claim information."""
    claim_number: str
    policy_number: str
    claimant_name: str
    date_of_loss: str
    loss_description: str
    estimated_repair_cost: float
    vehicle_details: Optional[str] = None

class PolicyQueries(BaseModel):
    queries: List[str] = Field(
        default_factory=list,
        description="A list of query strings to retrieve relevant policy sections."
    )

class PolicyRecommendation(BaseModel):
    """Policy recommendation regarding a given claim."""
    policy_section: str = Field(..., description="The policy section or clause that applies.")
    recommendation_summary: str = Field(..., description="A concise summary of coverage determination.")
    deductible: Optional[float] = Field(None, description="The applicable deductible amount.")
    settlement_amount: Optional[float] = Field(None, description="Recommended settlement payout.")

class ClaimDecision(BaseModel):
    claim_number: str
    covered: bool
    deductible: float
    recommended_payout: float
    notes: Optional[str] = None

# 5. Define Tools
@tool
def parse_claim(file_path: str) -> str:
    """
    Parse a claim JSON file and return structured ClaimInfo data.

    Args:
        file_path (str): Path to the JSON file containing claim data.
    """
    print("[INSIDE TOOL]: parse_claim")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        claim_info = ClaimInfo.model_validate(data)
        return claim_info.model_dump_json()
    except Exception as e:
        return f"Error parsing claim: {str(e)}"

@tool
def is_valid_query(query: str) -> bool | str:
    """
    Check if the claim made by the user is valid and return a boolean value, with reason.
    Args:
        query (str): The parsed text in json format from the parser tool with all the information.
    """
    print("[INSIDE TOOL]: is_valid_query")
    try:
        claim_info = ClaimInfo.model_validate_json(query)
        coverage_data = []
        with open("coverage_data.csv", "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                coverage_data.append(row)
    except Exception as e:
        return f"Error : {str(e)}"

    policy = next(
        (p for p in coverage_data if p["policy_number"] == claim_info.policy_number),
        None
    )

    if policy is None:
        return (False, "Policy not found.")

    dues_remaining = policy.get("claim_dues_remaining", "").strip().lower()
    if dues_remaining in ("true", "1", "yes"):
        return (False, "Due to outstanding payments, the policy is considered invalid.")

    date1 = datetime.datetime.strptime(str(claim_info.date_of_loss), "%Y-%m-%d")
    date2 = datetime.datetime.strptime(str(policy["coverage_start_date"]), "%Y-%m-%d")
    date3 = datetime.datetime.strptime(str(policy["coverage_end_date"]), "%Y-%m-%d")

    if not (date2 <= date1 <= date3):
      return (False, "The date of loss falls outside the policy’s coverage period.")

    return (True, "Valid claim.")

@tool
def generate_policy_queries(claim_info_json: str) -> str:
    """
    Generate queries to retrieve relevant policy sections based on claim info.

    Args:
        claim_info_json (str): JSON string of ClaimInfo data.
    """
    print("[INSIDE TOOL]: generate_policy_queries")
    prompt = f"""
    Analyze the following auto insurance claim to identify 3-5 key policy sections to consult:
    - Focus on collision coverage, liability, deductibles, and relevant exclusions or endorsements.
    - Claim Data: {claim_info_json}
    - Return a JSON object with a 'queries' field containing a list of strings, e.g., {{"queries": ["query1", "query2", "query3"]}}. Do not include metadata fields.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = model(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            try:
                result = json.loads(response_content)
                if isinstance(result, dict) and "queries" in result:
                    queries = result["queries"]
                    if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                        return json.dumps(result)
                    elif isinstance(queries, list):
                        queries = [q["query"] if isinstance(q, dict) and "query" in q else str(q) for q in queries]
                        return json.dumps({"queries": queries})
                return json.dumps({"queries": []})
            except json.JSONDecodeError:
                return f"Error: Invalid JSON response from model: {response_content}"
        except Exception as e:
                return f"Error generating policy queries: {str(e)}"

@tool
def retrieve_policy_text(queries_json: str) -> str:
    """
    Retrieve policy text from ChromaDB based on queries.

    Args:
        queries_json (str): JSON string of PolicyQueries.
    """
    print("[INSIDE TOOL]: retrieve_policy_text")
    try:
        queries_data = json.loads(queries_json)
        if not isinstance(queries_data, dict) or "queries" not in queries_data:
            return f"Error: Invalid queries_json format, expected {{'queries': [...]}}, got {queries_json}"

        queries = queries_data["queries"]
        if not isinstance(queries, list):
            return f"Error: Queries field must be a list, got {type(queries)}"

        query_strings = []
        for q in queries:
            if isinstance(q, dict) and "query" in q:
                query_strings.append(q["query"])
            elif isinstance(q, str):
                query_strings.append(q)

        queries = PolicyQueries(queries=query_strings)
        policy_texts = []
        for query in queries.queries:
            query_embedding = sentence_transformer_ef(query)[0]
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )
            relevant_chunks = results['documents'][0]
            policy_texts.extend(relevant_chunks)

        return "\n\n".join(policy_texts)
    except json.JSONDecodeError:
        return f"Error: Invalid JSON in queries_json"
    except Exception as e:
        return f"Error retrieving policy text: {str(e)}"

@tool
def generate_recommendation(claim_info_json: str, policy_text: str) -> str:
    """
    Generate a policy recommendation based on claim info and retrieved policy text.

    Args:
        claim_info_json (str): JSON string of ClaimInfo data.
        policy_text (str): Retrieved policy text.
    """
    print("[INSIDE TOOL]: generate_recommendation")
    prompt = f"""
    Evaluate the following auto insurance claim against the policy text:
    - Determine if the collision is covered, the deductible, settlement amount, and applicable policy section.
    - Claim Info: {claim_info_json}
    - Return a JSON object matching the following schema:
      {{
        "policy_section": "str", // The specific policy section or clause (e.g., 'Exclusions', 'Collision Coverage')
        "recommendation_summary": "str", // Concise summary of coverage determination
        "deductible": float or null, // Applicable deductible amount, if any
        "settlement_amount": float or null // Recommended payout, if any
      }}
    - Example:
      {{
        "policy_section": "Exclusions",
        "recommendation_summary": "Claim denied due to business use exclusion",
        "deductible": null,
        "settlement_amount": 0.0
      }}
    - Do not use fields like 'recommendation', 'coverage_evaluation', or 'reason'.
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = model(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            try:
                result = json.loads(response_content)
                PolicyRecommendation.model_validate(result)
                return response_content
            except json.JSONDecodeError:
                return f"Error: Invalid JSON response from model: {response_content}"
            except Exception as e:
                return f"Error: Invalid recommendation format: {str(e)}"
        except Exception as e:
                return f"Error generating recommendation: {str(e)}"

@tool
def finalize_decision(claim_info_json: str, recommendation_json: str) -> str:
    """
    Finalize the claim decision based on the recommendation.

    Args:
        claim_info_json (str): JSON string of ClaimInfo data.
        recommendation_json (str): JSON string of PolicyRecommendation.
    """
    print("[INSIDE TOOL]: finalize_decision")
    try:
        claim_info = ClaimInfo.model_validate_json(claim_info_json)
        rec_data = json.loads(recommendation_json)
        if not isinstance(rec_data, dict):
            return f"Error: recommendation_json must be a JSON object, got {type(rec_data)}"

        if "policy_section" not in rec_data:
            recommendation_text = rec_data.get("recommendation", rec_data.get("reason", rec_data.get("coverage_evaluation", "")))
            policy_match = re.search(r'(Part [A-D]|\bExclusions\b|\bCollision Coverage\b)', recommendation_text, re.IGNORECASE)
            rec_data["policy_section"] = policy_match.group(0) if policy_match else "Unknown Section"
        if "recommendation_summary" not in rec_data:
            rec_data["recommendation_summary"] = rec_data.get("recommendation", rec_data.get("reason", rec_data.get("coverage_evaluation", "No summary provided")))
        if "deductible" not in rec_data:
            rec_data["deductible"] = None
        if "settlement_amount" not in rec_data:
            rec_data["settlement_amount"] = 0.0

        rec = PolicyRecommendation.model_validate(rec_data)
        covered = "covered" in rec.recommendation_summary.lower() or (rec.settlement_amount is not None and rec.settlement_amount > 0)
        deductible = rec.deductible if rec.deductible is not None else 0.0
        recommended_payout = rec.settlement_amount if rec.settlement_amount else 0.0
        decision = ClaimDecision(
            claim_number=claim_info.claim_number,
            covered=covered,
            deductible=deductible,
            recommended_payout=recommended_payout,
            notes=rec.recommendation_summary
        )
        return decision.model_dump_json()
    except Exception as e:
        return f"Error finalizing decision: {str(e)}"

# 6. Prompts & Agent Setup
system_prompt = (
    """
    You are an expert insurance claim-processing agent specializing in
    auto insurance. You follow a strict, multi-step reasoning process.

    CLAIM PROCESSING ORDER (MANDATORY):

      1. Parse the claim JSON to extract all ClaimInfo fields.

      2. Validate the claim:
         - Pass ClaimInfo to the claim-validity function.
         - If the result is False, STOP immediately and return an
           invalid-claim decision.

      3. Generate policy-related search queries based on the extracted claim details.

      4. Retrieve relevant policy text from ChromaDB using the generated queries.

      5. Use the web-search tool to estimate typical repair costs:
         - Search for market repair price ranges for the described damage type.
         - Compare estimated cost to the claimed amount.
         - If the claimed amount is unrealistic or inflated,
           issue an invalid-claim decision.

      6. Generate a recommendation based on:
         - validated claim details,
         - retrieved policy text,
         - estimated repair cost.

      7. Produce the final claim decision, including coverage status,
         deductible, and recommended payout.

    ALWAYS follow this exact sequence. Do not reorder, skip, or combine steps.
    """
)

planning_prompts = PlanningPromptTemplate(
    initial_facts=(
        """
        Claim details:
        {claim_info_json}

        Policy details:
        {policy_text}

        Recommendation (if any):
        {recommendation_json}
        """
    ),
    initial_plan=(
        """
        1. Parse the claim JSON and extract all ClaimInfo fields.

        2. Validate the claim:
           - Call the claim-validity function with the extracted ClaimInfo.
           - If False, STOP and return an invalid-claim response.

        3. Generate policy search queries based on the claim details.

        4. Retrieve relevant policy sections from ChromaDB using those queries.

        5. Estimate repair cost:
           - Use the web-search tool to research typical repair prices for
             the specific type of damage.
           - Compare estimated market price with the claimed amount.
           - If the claim amount is unreasonable or inflated,
             return an invalid-claim result.

        6. Using claim details, policy text, and repair-cost estimate,
           create a coverage recommendation.

        7. Produce the final claim decision in the required output format.
        """
    ),
    update_facts_pre_messages="Reassess claim and policy facts with new information:",
    update_facts_post_messages="The facts have been updated.",
    update_plan_pre_messages="Revise the plan based on updated facts:",
    update_plan_post_messages="Plan updated."
)

managed_agent_prompts = ManagedAgentPromptTemplate(
    task=(
        "Process the following auto insurance claim: {task_description}. "
        "Follow the structured claim-processing workflow exactly as defined."
    ),
    report=(
        "Generate the final claim decision using the processed results: {results}"
    )
)

final_answer_prompts = FinalAnswerPromptTemplate(
    pre_messages="Summarize the claim decision based on the full reasoning process.",
    post_messages="Ensure the final decision is clear, concise, and well-structured.",
    final_answer_template=(
        """
        Claim Decision
        -------------------------
        Claim Number: {claim_number}
        Covered: {covered}
        Deductible: {deductible}
        Recommended Payout: {recommended_payout}

        Notes:
        {notes}
        """
    )
)

prompt_templates = PromptTemplates(
    system_prompt=system_prompt,
    planning=planning_prompts,
    managed_agent=managed_agent_prompts,
    final_answer=final_answer_prompts
)

claim_processing_agent = ToolCallingAgent(
    tools=[parse_claim, is_valid_query, WebSearchTool(), generate_policy_queries, retrieve_policy_text, generate_recommendation, finalize_decision],
    model=model,
    add_base_tools=True,
    prompt_templates=prompt_templates
)

# 7. Flask Server setup
app = Flask(__name__)

def strip_ansi(text: str) -> str:
    """Helper to remove ANSI escape sequences from captured logs."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/process_claim', methods=['POST'])
def process_claim():
    try:
        claim_data = request.json
        temp_file = 'temp_claim.json'
        with open(temp_file, 'w') as f:
            json.dump(claim_data, f)
        
        # Capture logs
        f_buf = io.StringIO()
        with contextlib.redirect_stdout(f_buf):
            try:
                agent_result = claim_processing_agent.run(temp_file)
            except Exception as inner_e:
                agent_result = f"Error during agent run: {str(inner_e)}"
        
        raw_logs = f_buf.getvalue()
        clean_logs = strip_ansi(raw_logs)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        # Parse decision JSON if present in result, or construct default
        parsed_decision = None
        
        # Search for any valid ClaimDecision structure in stdout logs or result
        decision_match = re.findall(r'(\{"claim_number".*?"notes":.*?\})', clean_logs)
        if decision_match:
            try:
                parsed_decision = json.loads(decision_match[-1])
            except Exception:
                pass
        
        if not parsed_decision:
            # Fallback parsing of agent_result if it can't find a JSON block
            covered_status = "covered: true" in agent_result.lower() or "covered: yes" in agent_result.lower()
            claim_num = claim_data.get("claim_number", "UNKNOWN")
            
            payout_match = re.search(r'recommended payout:\s*\$?([\d\.]+)', agent_result, re.IGNORECASE)
            payout = float(payout_match.group(1)) if payout_match else 0.0
            
            deduct_match = re.search(r'deductible:\s*\$?([\d\.]+)', agent_result, re.IGNORECASE)
            deductible = float(deduct_match.group(1)) if deduct_match else 0.0
            
            parsed_decision = {
                "claim_number": claim_num,
                "covered": covered_status,
                "deductible": deductible,
                "recommended_payout": payout,
                "notes": agent_result.strip()
            }

        return jsonify({
            "success": True,
            "decision": parsed_decision,
            "logs": clean_logs
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
