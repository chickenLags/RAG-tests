

# Introduction
In this repository I'm learning about RAG's and llms. The first and 
current project is the attempt to use mistral in conjunction with 
langchain/FAISS. To get answers based on a folder containing pdf's.

Specifically, these books should contain specific discipline knowledge,
which can be queried when needed. 


## How to start (Mac):
Download ollama and mistral:
```bash
brew install ollama
ollama serve
ollama run mistral (4.7GB) # Might need to be ran before it works. I'm not sure.
```

Install and run project:
```bash
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## State of project:
Currently the retrieval of embeddings is not good at all and should 
be optimized.
