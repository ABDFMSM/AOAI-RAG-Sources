from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureOpenAI
from langchain.tools import Tool
from dotenv import load_dotenv
import os, sys

#Load env variable and configure the embedding model
load_dotenv()

# Here we have a vectorDB_tool class that will return the tool used for chatbot agent. 
class vectorDB_tool:
    def __init__(self):
        self.embedding_function = AzureOpenAIEmbeddings(azure_deployment=os.getenv("Embedding_model"))
    
    def load_pages(self):
        """
        Load and split the specified PDF files into pages, and save the vector embedding database on the local storage.

        Parameters:
        - self: The vectorDB_tool instance.
        - paths (list): A list of file paths to the PDF files.

        Returns:
        None
        """
        paths = list(sys.argv[1:])
        for path in paths:
            # Load the file and split it into pages. 
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            # Here we are saving our vector embedding database on the local storage. 
            Chroma.from_documents(pages, self.embedding_function, persist_directory=r".\chroma_db")

    def vectorDB_loader(self):
        """
        vectorDB_loader function is used to load the vector embedding database from the local storage. 
        """
        db = Chroma(persist_directory="./chroma_db", embedding_function=self.embedding_function)
        retriever = db.as_retriever()
        chain = RetrievalQA.from_chain_type(
            llm=AzureOpenAI(deployment_name=os.getenv("Completion_model"), 
            openai_api_version="2024-02-15-preview"), 
            chain_type='stuff', 
            retriever=retriever, 
            return_source_documents=True, 
            input_key="question")
        return chain
    
    def return_tool(self):
        tool = Tool(
            name="NorthWind_Insurance",
            func=lambda query: self.vectorDB_loader().invoke({"question": query}),
            description="Use it to answer any questions related to insurance."
        )
        return tool
