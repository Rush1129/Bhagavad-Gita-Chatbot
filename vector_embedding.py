import json
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import sys

load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

with open('bhagavad_gita.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

documents = []

for chapter, verses in data.items():
    for verse_num, content in verses.items():
        text = (
            f"Translation: {content.get('translation', '')}\n"
            f"Purport: {content.get('purport', '')}"
        )
        metadata = {'chapter':chapter,
         'verse': verse_num,
         'sanskrit': content.get('sanskrit', ''),
         'translation': content.get('translation', ''),
         'purport': content.get('purport', '')}

        documents.append(Document(page_content=text, metadata=metadata))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory='gita_vectordb')