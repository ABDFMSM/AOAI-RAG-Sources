from tools.SQLTool import SQLLoader
from tools.BingTool import return_tools
from tools.ChromaDBTool import vectorDB_tool
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

# Configure system message and conversation template.      
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an AI assistance who can access 3 tools: 
            1) NorthWind_Insurance tool which is a vector Database to get answers to customer's questions regarding insurance:
            Stick to the information provided by the database. 
            If you don't find an answer to the question, then use bing_search tool.
            Provide a brief one paragraph answer to the question unless the user asks you to list things.  
            Return the source documents in this format: Source: {document name} in {pages} pages.
            2) An Azure SQL Database to get answers to customer's questions:
            In Azure SQL use 'TOP' keyword to limit number of results returned.  
            You mainly have Titanic and Books tables that you can use to answer user's questions. 
            Check the table schema when you don't have enough information to answer the question.
            Always return the SQL command that you used to perform your query. 
            3) bing_search tool:
            The bing_search tool returns the webpage content that contains information that you use to answer user's question. 
            Whenever asked about time and date just provide a short answer. 
            For other questions provide a max of one paragraph unless instructed otherwise.
            Always provide the link in the following format. "\nUsed this link: {link} to answer your question. 
            """
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        MessagesPlaceholder(
            variable_name='agent_scratchpad'
        ),  # where tools are loaded for intermediate steps.
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # Where the human input will injected
    ]
)

# Configuring the required tools (Azure SQL Tool, ChromaDB, BingSearch) 
tool_sql = SQLLoader().return_tool()
tool_vector = vectorDB_tool().return_tool()
tool_bing = return_tools()

# Configuring the requried components for the chatbot (tools, memory, llm_model)
tools_list = tool_sql + [tool_vector] + tool_bing
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k= 8)
llm = AzureChatOpenAI(deployment_name=os.getenv("Chat_deployment"), temperature=0)

agent = create_openai_tools_agent(llm, tools_list, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools_list,
    verbose=True,
    memory=memory, 
    max_iterations= 8 # Number of tries to retrieve data before exiting agent. 
)

def main():
    question = input("What do you like to ask?\n")
    while "exit" not in question.lower():  
        answer = agent_executor.invoke({"input": question})
        print(answer['output'])  
        question = input("\nDo you have other queries you would like to know about? if not type exit to end the chat.\n")  
    print(memory.load_memory_variables({})) #print the chat history. 

if __name__ == "__main__":
    main()