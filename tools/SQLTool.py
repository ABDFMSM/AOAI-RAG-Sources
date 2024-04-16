from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit

load_dotenv()

class SQLLoader:    
    def __init__(self, df=None):    
        self.engine = self.create_engine()  
        self.df = df  
  
    def create_engine(self):  
        username = os.getenv('SQL_USERNAME')  
        password = os.getenv('SQL_PASSWORD')  
        server = f"tcp:{os.getenv('SQL_ENDPOINT')}"
        database = os.getenv('SQL_DATABASE')  
        driver = "ODBC Driver 18 for SQL Server"  
        connection_string = f"mssql+pyodbc:///?odbc_connect=DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"  
        engine = create_engine(connection_string)  
        return engine   

    def create_table(self, table_name):  
        # Use SQLAlchemy to generate the table schema from the dataframe  
        self.df.to_sql(table_name, self.engine, if_exists='replace', index=False)  
        print(f"Table {table_name} has been created in the database.")
      
    def insert_data(self, table_name):  
        # Use pandas to_sql method for inserting the data  
        self.df.to_sql(table_name, self.engine, if_exists='append', index=False)  
        print(f"Data has been inserted into {table_name} table.")

    def read_db(self):
        return SQLDatabase(self.engine)
    
    def return_tool(self):
        sql_db = self.read_db()
        llm = AzureChatOpenAI(deployment_name=os.getenv("Chat_deployment"), temperature=0)
        toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
        return toolkit.get_tools()


