from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# We define a simple Python class which will specify the type of content that we want our LLm to generate
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

llm = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
# In Python, a parser is a component or function that analyzes input data (like text, JSON, XML, or HTML) 
# and converts it into a structured format that Python can easily work with


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Step 1: We define the output format using a Pydantic model defined in the class ReseachResponse.
# This model specifies the structure of the output we want from the LLm, including fields like topic, summary, sources, and tools.  

# # Step 2: We define the type of output we want via the parser (which is basically a string)
# With parser.get_format_instructions we take the instructions defined in the pydantic model 
# and turn them into a string that we can then give to the LLm as part of the prompt

# Step 3: We take the output of the LLm and with the parser we turn it into a Python object


tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#we set verbose to True so that we can see the thought process of the LLm and the tools it uses
query = input("What do you want to research?")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
 
# The output is expected to contain a list of dictionaries, and the relevant information is in the first element of that list
# Each dictionary in that list represents a step in the agent's thought process, including the final output.
# The text key within the first dictionary is where the actual string containing the research response is located.
# The parser then takes this string and converts it into a structured Python object based on the ResearchResponse model.
# This allows us to easily access the structured data like topic, summary, sources, and tools used in the research.

# In essence, raw_response.get("output") might yield something like [{"text": "...", "other_info": "..."}, {"text": "...", ...}], 
# and we are interested in the "text" field of the very first item in that list.