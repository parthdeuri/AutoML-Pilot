import json
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import PipelineState

def report_node(state: PipelineState) -> dict:
    """
    Invokes the LLM to write the final Markdown Model Card.
    """
    with open("agent/prompts/report_prompt.txt", "r") as f:
        system_prompt = f.read()
        
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    
    context = f"""
    Data Profile: {json.dumps(state['data_profile'], indent=2)}
    Cleaning Plan Executed: {json.dumps(state['cleaning_plan'], indent=2)}
    Model Metrics: {json.dumps(state['model_metrics'], indent=2)}
    Best Model Selected: {json.dumps(state['best_model_info'], indent=2)}
    """
    
    model_errors = state.get("model_errors")
    if model_errors:
        context += f"Models That Failed To Train (mention in the report): {json.dumps(model_errors, indent=2)}"
    
    messages = [
        ("system", system_prompt),
        ("human", f"Based on the following execution trace, write the final Model Card:\n{context}")
    ]
    
    response = llm.invoke(messages)
    
    content = response.content
    if isinstance(content, list):
        content = "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    elif not isinstance(content, str):
        content = str(content)
    
    return {"report": content}
