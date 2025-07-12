from autogen import  AssistantAgent
from config.llm import llm_config
fetch_data_agent = AssistantAgent(
    name="fetch_data_agent",
    system_message="""
    You are helpful data-fetching agent. Your task is to get data using the provided SQL.
    Follow the plan Strictly:
    1. Get the new SQL. 
    2. Call fetch_data function and pass the SQL, fetch the data, and return it in a JSON structured format.
    3. If output is 'No data found.' TERMINATE RAISE ERROR
    4. TERMINATE""",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

