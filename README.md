# BES4RAG: Best Embedding Selection for Retrieval-Augmented Generation

![Pipeline Diagram](pipeline.png)

## Overview
This framework selects the optimal embedding model to be paired with a large language model (LLM) in a Retrieval-Augmented Generation (RAG) system. The system is designed to answer questions based on a given knowledge source, which can be in the form of PDFs, text files, or transcripts from audio and video sources.

A key component of this approach is an automated multiple-choice question generation module, supported by a questioning LLM. Once the questions are generated, documents are segmented into chunks, embeddings are precomputed, and the most relevant segments are retrieved to assist the answering LLM in answering the questions. The system evaluates both the accuracy of the responses and the quality of the retrieved document segments, enabling the selection of the best embedding model for a given dataset.

The system has been successfully tested with datasets containing up to **50,000 documents**, demonstrating scalability across both **CPU** and **GPU** environments.

## Time Considerations  

The time required to compute embeddings depends on the available hardware:  
- **GPU:** Processing can take approximately **1 hour** for medium-sized datasets and extend to several hours for very large datasets.  
- **CPU:** The embedding computation time may range from **1 to 2 days**, depending on the number of documents and the complexity of the selected embedding model.  

Since the framework currently integrates LLM providers such as **Groq, OpenAI, and Gemini**, response times for question answering can also vary based on several factors:  
- The number of generated questions.  
- API rate limits, especially for free-tier users, which may impose daily restrictions on query processing speed.  

## How to run the pipeline? 


There are two ways to test the framework:

1. **Google Colab**: Run the framework directly in Google Colab by following this link and executing the code step by step. The notebook includes detailed instructions on setting up your datasets and experiments. Using a GPU is highly recommended.  
   [Run in Google Colab](https://colab.research.google.com/drive/1VGMA1cHQ2ClTuXvKgVDcqhYA14YRgEX2?usp=sharing)
   
2. **Local Machine**: Alternatively, you can run the framework on your local machine by following the setup instructions provided later in this README.


## Installation

Clone this repository then choose one of the following options.

### Option 1 - Use Docker
You can use the `Dockerfile` to create the environment for running the framework. This method ensures a consistent environment and makes it easier to set up on any machine. To build and run the Docker container, use the following commands:

```bash
# Build the Docker image
docker build -t BES4RAG .

# Run the Docker container
docker run -it --rm BES4RAG
```

### Option 2 - Manual installation
```bash
python setup.py install
```

After cloning the repository, you can either use the Dockerfile to create the environment or install it manually. In the case of manual installation, there may be fewer guarantees regarding environment consistency and dependencies.


## API Key Configuration  

To use this code, you must provide the necessary API keys by filling in the `credentials/api_keys.json` file.  

Only the required keys need to be set; you can remove any unused ones.  

#### Example:  

If you are using OpenAI, your `credentials/api_keys.json` should look like this:  

```json
{
    "openai": {"api_key":"your-openai-api-key"}
}
```

If you are using both OpenAI and Groq:  

```json
{
    "openai": {"api_key":"your-openai-api-key"},
    "groq": {"api_key":"your-groq-api-key"}
}
```  



### Recommended API Key Setup  

The **default version of this pipeline uses both Groq and OpenAI**, so it is **highly recommended** to set API keys for both services.  

Make sure not to share your API keys publicly.


## Dataset Configuration  

To set up the dataset, follow these steps:  

1. **Create a folder** named as your dataset.  
2. **Inside this folder, create a subfolder named `files`** and place all the files you want to process there. The supported formats include `.json`, `.pdf`, and `.txt`.  
3. **In the main dataset folder, add a CSV file named `embedding_models.csv`** to specify the embedding models you want to use.  

#### Example Folder Structure:  
```
/my_dataset
    /files
        file1.json
        file2.pdf
        file3.txt
    embedding_models.csv
```

#### CSV Format (`embedding_models.csv`):  
The file should follow this format, listing the embedding models you intend to use:  

```
Type;Model Name
sentencetransformers;sentence-transformers/all-MiniLM-L6-v2
sentencetransformers;intfloat/multilingual-e5-large
sentencetransformers;dunzhang/stella_en_1.5B_v5
colbert;antoinelouis/colbert-xm
openai512;text-embedding-3-large
openai;text-embedding-3-large
```

Ensure that all necessary models are listed correctly.

### Difference Between `openai` and `openai512`  

- **`openai`** uses the **full token input length** supported by the model (e.g., over 8000 tokens). This allows for a broader context window but can introduce **issues with high `k` values** (number of retrieved passages in retrieval-augmented generation, RAG). 
- **`openai512`** **limits the input to 512 tokens**, making it more comparable to standard `sentence-transformers` models. This restriction helps **ensure better stability** and makes embeddings more **consistent** across different models.  

If you're working with RAG, it's **recommended to use `openai512`** for better performance and comparability with other embedding models.  

### Customizing the Embedding Models  

You can **remove or add models** as needed, as long as they belong to one of the following supported types:  

- **`colbert`**  
- **`sentencetransformers`**  
- **`openai`**  
- **`openai512`**  

More model types will be supported in future updates.

## Usage

Once installed, the entire pipeline can be launched using:
```bash
python launch.py <dataset_path>
```
where <dataset_path> is the path to the dataset containing the knowledge source.


