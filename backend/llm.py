from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import pandas as pd


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text


    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class LLM:
    def __init__(self, vector_store, file_path):
        self.load_excel_data(file_path)
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o",
            streaming=True
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.setup_tools()
        self.setup_agent()


    def translate_to_lithuanian(self, text):
        translation_prompt = f"""Translate the following text to Lithuanian. 
        If the text is already in Lithuanian, return it as is:

        {text}

        Lithuanian translation (translation is mandatory!):"""
        
        messages = [HumanMessage(content=translation_prompt)]
        translated = self.llm.predict_messages(messages).content
        return translated


    def is_relevant_question(self, question):
            relevance_prompt = f"""Determine if the following question is related to "Smiltynės perkėla" or ferry services:

            Question: {question}

            Answer with 'Yes' if it's related, or 'No' if it's not related."""

            messages = [HumanMessage(content=relevance_prompt)]
            response = self.llm.predict_messages(messages).content.strip().lower()
            return response == 'yes'


    def load_excel_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['laikas'] = pd.to_datetime(self.df['laikas'])

    def setup_tools(self):
        python_repl = PythonAstREPLTool(locals={"df": self.df})
        python_repl_tool = Tool(
            name='Python_REPL',
            func=python_repl.run,
            description='Use this tool when you need to analyse transport flows with pandas using the df data frame. The "laikas" column is already parsed to datetime format, the "praleidimas" is an integer indicating how many cars passed that hour. Make sure your answer is structured.'
        )

        # DuckDuckGo Search Tool
        search = DuckDuckGoSearchRun()

        duckduckgo_tool = Tool(
            name='DuckDuckGo_Search',
            func=lambda query: search.run(f"Smiltynės perkėla keltas.lt {query}"),
            description='Use this tool if you need additional information about "Smiltynės perkėla" from the Internet. Prompt this knowledge base in Lithuanian. Make sure your answer is structured.'
        )

        retriever_tool = Tool(
            name='Knowledge_Base',
            func=lambda query: '\n'.join([doc.page_content for doc in self.vector_store.similarity_search(query, k=3)]),
            description='Prompt this knowledge base in lithuanian. Use it to retrieve information from the Smiltynė ferry knowledge base. Make sure your answer is structured.'
        )

        self.tools = [retriever_tool, python_repl_tool, duckduckgo_tool]
    
    def setup_agent(self):
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=7,
            max_execution_time=30,
            early_stopping_method="generate",
            memory=self.memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": '''You are an expert on "Smiltynės perkėla" ferry services you MUST answer related only to that. 
                You have tools from their webpage (Knowledge_Base), the internet (DuckDuckGo_Search) and dataframe with traffic data (Python_REPL).
                Current date 2024-10-17.

                Instructions:
                1. Always provide informative and helpful responses.
                2. If the question is not related to "Smiltynės perkėla" or ferry services politely redirect to the topic.
                3. ALWAYS use the Knowledge_Base tool first. Only if it doesn't provide sufficient information, consider using other tools.
                4. When using the Python_REPL tool, ALWAYS provide the exact Python code as the Action Input.
                5. On the last iteration, ALWAYS provide a final answer without using tools.
                6. Format your response EXACTLY as shown below.

                Human: {input}
                AI: Certainly! I'll do my best to answer your question about Smiltynės perkėla ferry services.

                Thought: [your thought process]
                Action: [tool name]
                Action Input: [For Python_REPL, provide ONLY the Python code. For other tools, provide the input query.]
                Observation: [result from tool]

                (Repeat Thought/Action/Action Input/Observation as needed, max 3 times)

                Thought: I now have enough information to provide a final answer.
                Action: Final Answer
                Action Input: [your final answer in English]

                Human: Thank you for the information. Do you have any more questions about Smiltynės perkėla or ferry services?

                Available tools:''',
                "suffix": "Begin!"
            }
        )
        
    def ask(self, question, stream_handler=None):
        callbacks = [stream_handler] if stream_handler else None
        
        try:
            response = self.agent_executor.run(
                input=question,
                chat_history=self.memory.chat_memory.messages,
                callbacks=callbacks
            )
            
            if "Action: Final Answer" in response:
                final_answer = response.split("Action: Final Answer")[-1].strip()
                final_answer = final_answer.split("Action Input:")[-1].strip()
            else:
                final_answer = response.strip()
            
            lithuanian_response = self.translate_to_lithuanian(final_answer)
            
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(lithuanian_response)
            
            if stream_handler:
                stream_handler.on_llm_new_token("\nFinal Answer: " + lithuanian_response)
            
            return lithuanian_response
        except Exception as e:
            error_message = f"An error occurred: {str(e)}. Please try rephrasing your question."
            if stream_handler:
                stream_handler.on_llm_new_token("\nError: " + error_message)
            return error_message