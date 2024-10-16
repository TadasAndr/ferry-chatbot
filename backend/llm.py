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
        temperature=0.3,
        model_name="gpt-4o",
        streaming=True
        )
        self.memory = ConversationBufferMemory(return_messages=True)
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
            description='Use this tool when you need to analyse transport flows with pandas using the df data frame. The "laikas" column is in date/time format, the "praleidimas" is an integer indicating how many cars passed that hour.'
        )

        # DuckDuckGo Search Tool
        search = DuckDuckGoSearchRun()
        duckduckgo_tool = Tool(
            name='DuckDuckGo_Search',
            func=search.run,
            description='Use this tool if you need additional information about "Smiltynės perkėla" from the Internet..'
        )

        retriever_tool = Tool(
            name='Knowledge_Base',
            func=lambda query: '\n'.join([doc.page_content for doc in self.vector_store.similarity_search(query, k=3)]),
            description='Priority to this tool. Use it to retrieve information from the Smiltynė ferry knowledge base.'
        )

        self.tools = [python_repl_tool, duckduckgo_tool, retriever_tool]
    
    def setup_agent(self):
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": '''You are an expert on "Smiltynės perkėla" ferry services.

            Instructions:
            1. Always provide informative and helpful responses.
            2. If the question is not related to "Smiltynės perkėla" or ferry services politely redirect to the topic.
            3. Use tools to gather information before answering if necessary.
            4. On iteration 3 ALWAYS have an answer!!!
            5. ALWAYS format your response exactly as shown below, including the Action and Action Input fields.
            
            Format your response as follows:

                Thought: [your thought]
                Action: [tool name]
                Action Input: [input for the tool]
                Observation: [result]
                ... (repeat as needed, max 3 times)
                Thought: I now know the final answer
                Final Answer: [your final answer in Lithuanian]

            Available tools:''',
                "suffix": '''Begin!

            Human: {input}
            {agent_scratchpad}'''
            }
        )
        
    def ask(self, question, stream_handler=None):
        # if not self.is_relevant_question(question):
        #     irrelevant_response = 'Atsiprašau, bet šis klausimas nėra susijęs su „Smiltynės perkėla" ar keltų paslaugomis. Ar galėčiau jums padėti su klausimu apie keltų paslaugas?'
        #     if stream_handler:
        #         stream_handler.on_llm_new_token(irrelevant_response)
        #     return irrelevant_response

        callbacks = [stream_handler] if stream_handler else None
        english_response = self.agent_executor.run(question, callbacks=callbacks)
        lithuanian_response = self.translate_to_lithuanian(english_response)
        
        # Send the translated response through the stream handler
        if stream_handler:
            stream_handler.on_llm_new_token("\nFinal Answer: " + lithuanian_response)
        
        return lithuanian_response