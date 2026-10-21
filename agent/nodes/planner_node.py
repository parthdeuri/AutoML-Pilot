import json
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import PipelineState

def planner_node(state: PipelineState) -> dict:
    """
    Invokes the LLM to generate a cleaning plan based on the data profile.
    """
    profile = state["data_profile"]
    
    with open("agent/prompts/planner_prompt.txt", "r") as f:
        system_prompt = f.read()
        
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    messages = [
        ("system", system_prompt),
        ("human", f"Here is the data profile:\n{json.dumps(profile, indent=2)}")
    ]
    
    response = llm.invoke(messages)
    
    # Parse JSON from response
    content = response.content
    if isinstance(content, list):
        content = "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    elif not isinstance(content, str):
        content = str(content)
        
    content = content.strip()
    # Clean up possible markdown code blocks if the LLM ignores instructions
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
        
    try:
        cleaning_plan = json.loads(content)
    except json.JSONDecodeError:
        # Fallback empty plan if parsing fails (the feature step still imputes/encodes safely)
        print("WARNING: Could not parse the cleaning plan from the LLM response; using an empty plan.")
        cleaning_plan = {"actions": []}
        
    return {"cleaning_plan": cleaning_plan}
