from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class LLM:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o",
            streaming=True
        )
        self.setup_chain()

    def setup_chain(self):
        prompt_template = """You are an expert of "Smiltynės perkela" ferry services page content.
        Based on given knowledge from the context (the context is from "Smiltynės perkela" webpage) helpfully answer user's question on "Smiltynės perkela" page content and ferry services.

            Instructions:
            1. Always respond in Lithuanian.
            2. If the question is unclear or lacks necessary information, ask for clarification.
            3. If the question is outside your expertise of given content or not related to "Smiltynės perkela" page content, politely redirect the conversation back to it.
            4. If you don't know say that you don't know.

            Context: {context}

            Question: {question}

            Answer (in Lithuanian):"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, question, session_id, stream_handler=None):
        callbacks = [stream_handler] if stream_handler else None
        response = self.chain({"query": question}, callbacks=callbacks)
        return response['result']