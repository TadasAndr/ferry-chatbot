from langchain.document_loaders import PyPDFLoader


def load_document(file):
    loader = PyPDFLoader(file)
    data = loader.load()
    return data