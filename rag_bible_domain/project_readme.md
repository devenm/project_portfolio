
Hello! Welcome to my 'God's Will - RAG Project'

This project utilizes the technique of Retrieval Augmented Generation (RAG) to tune and optimize performance for an LLM model,
specifically ChatGPT. More specifically, it uses an interesting use case of optimization for the Bible. 

I have zero religious affiliation with the Bible, but I thought it would be cool to test a unique reality: 

"If we had a book, that held all the answers to life, existence, and purpose, what would you ask it?"


Note: RAG is a powerful method, and I believe that it would be better showcased on textual data where information is objective (like an instruction manual). However, I would argue that it is infinitely more fun to make a predictive model biased towards the content of this famous piece of text. 


How it works:
-The Bible gets split into chunks.
-These chunks are converted into vectors using OpenAIEmbeddings
-The chunks are stored locally in ChromaDB with their embedding as a sort-key
-A user asks a question/queries the OpenAI model
    -The question/query gets vectorized
    -Using a similarity metric, Chroma will extract the 3 most similar chunks from the database
    -The prompt is altered to include these three chunks, and they are described as "Context" to the LLM
    -ChatGPT recieves context, the question, and the command: Answer the following question based on the above context".
-The LLM now has specific information to pull from, leading to a more domain localized answer.


Skills showcased:
-Basic software development in Python
-Use of LangChain as a structure for LLM operations
-Optimization for LLMs through RAG
-Embedding operations
-Local database operations



Some Sources:
LangChain Syntax - https://python.langchain.com/docs 
Bible text file - https://bereanbible.com/bsb.txt 
Inspiration for this project - https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami