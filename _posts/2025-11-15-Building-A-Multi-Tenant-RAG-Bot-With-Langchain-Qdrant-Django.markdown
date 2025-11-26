---
layout: post
title:  "Building a Multi Tenant RAG Bot With Langchain, Qdrant and Django"
date:   2025-11-15 11:14:11 -0700
tags: [web-dev, llm] 
---

This was one of my hobby projects. I wanted to get close to working with LLMs and RAG, and this project taught me a lot of things. 

The GitHub Repo is here: [Multi Tenant Ragbot Github Repo](https://github.com/sushant021/multi-tenant-ragbot)

The goal of this post is to help developers understand the design decisions and implementation patterns required to build a production-grade RAG system that supports multiple users, each with isolated knowledge spaces and secure data boundaries.

Most RAG tutorials focus on simple, single-user workflows. Real-world systems need much more:

- User authentication and isolation
- Individual knowledge bases per user
- Namespace-level filtering (filespaces)
- User-defined system prompts
- Session-level conversation memory
- Secure document upload and storage
- Vector search with metadata
- Context retrieval + reranking
- LLM orchestration with history
- Clean deletion, rollback, and cleanup

This project integrates all of these requirements into a Django application with a complete RAG pipeline.

<h5 class="fw-bold mt-5"> 
Tools Used
</h5>
Below are the tools that I've used for this project. 

Web Framework: Django   
Raw Data Storage: AWS S3  
Session Storage / Conversation History Storage: Redis  
Vector Storage: Qdrant  
Database: PostgreSQL  
LLM: llama-3.3-70b-versatile, Powered by Groq [https://console.groq.com/docs/model/llama-3.3-70b-versatile](https://console.groq.com/docs/model/llama-3.3-70b-versatile)

<h5 class="fw-bold mt-5"> 
Workflow
</h5>
1. User Registration, Login and Authentication  
   A `CustomUser` model, extending Django's `AbstractUser`, is used to manage user accounts. All the registration and authentication process are handled by Django's native processes. 

2. Filespaces (Namespaces)    
    Each user can create multiple Filespaces (similar to Namespaces). Each Filespace allows multiple documents to be uploaded. I created Filespace thinking about different departments of the company needing access to different documents. 

3. Document upload   

    When a user uploads a document to any Filespace

    - it is uploaded to S3
    - temporarily downloaded for processing
    - parsed via `PyMuPDFLoader`
    - split using `RecursiveCharacterTextSplitter`
    - embedded using `FastEmbed BGE-small-en-v1.5`
    - build Qdrant points  
    - Upsert to Qdrant 

    Each Qdrant point has the vector embedding along with the following payload: 
    ``` python
    payload={  
        "user_id": doc.filespace.owner.id,  
        "filespace_id": doc.filespace.id,  
        "content": chunk.page_content,  
        "source_file_id": doc.id,  
        "description": doc.description,  
        "file_size": doc.file_size,  
        "file_type": doc.file_type,  
    }
    ```



4. Vector Search and Reranking

    When a user sends a chat message or the so-called query,
    - it is embedded using the same embedding model `FastEmbed BGE-small-en-v1.5`
    - search Qdrant with the payload filters 
    - return top hits
    - rerank the hits with CrossEncoder model
    - select the best context chunks
    - inject into LLM prompt
    - send the whole prompt to LLM  <br><br>

    Reranking implementation:   
    ```python
    reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    scores = reranker.rerank(query, hit_contents)
    ```


5. LLM Orchestration via Groq

    Groq's Llama-3.3 70B model is used for final generation.

    The message structure includes:

    - System prompt (user-defined)
    - Conversation history
    - Retrieved context
    - The new query

    ```python
    messages.append(SystemMessage(content=enhanced_system_prompt))
    messages.append(HumanMessage(content=current_query_with_context))
    response = llm.invoke(messages)
    ```

    The final result also gets added to Redis conversation memory.

6. Conversation Memory with Redis  

    Each chat session has its own Session ID.

    Redis stores:

    - Message roles (user or assistant)
    - Content
    - Timestamps  

    Utility functions include:
    - `get_conversation_history()`
    - `add_to_conversation_history()`
    - `create_new_session()`  
      
    Sessions have an expiration time for cleanup.

7. Document Deletion

    When a document is deleted:

    - All vectors from Qdrant are removed
    - File is removed from S3
    - Django model instance is deleted  

  
    Rollback logic ensures cleanup if failures occur.

<h5 class="fw-bold mt-5"> 
Conclusion
</h5>
Building a multi-tenant RAG system requires more than embedding a PDF and calling an LLM. Authentication, secure storage, vector isolation, prompt management, conversation memory, and robust deletion workflows all must come together. This project provides a complete and extensible foundation for building real-world RAG applications. Hope it helps !! 


