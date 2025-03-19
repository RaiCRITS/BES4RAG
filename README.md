# BES4RAG: Best Embedding Selection for Retrieval-Augmented Generation

![Pipeline Diagram](pipeline.png)

## Overview
This framework selects the optimal embedding model to be paired with a large language model (LLM) in a Retrieval-Augmented Generation (RAG) system. The system is designed to answer questions based on a given knowledge source, which can be in the form of PDFs, text files, or transcripts from audio and video sources.

A key component of this approach is an automated multiple-choice question generation module, supported by a questioning LLM. Once the questions are generated, documents are segmented into chunks, embeddings are precomputed, and the most relevant segments are retrieved to assist the answering LLM in answering the questions. The system evaluates both the accuracy of the responses and the quality of the retrieved document segments, enabling the selection of the best embedding model for a given dataset.

The system has been tested with datasets containing up to approximately 50,000 documents, both on CPU and GPU environments.


There are two ways to test the framework:

1. **Google Colab**: You can easily run the framework in Google Colab using the following link and following the code step by step (all the instructions to set up your datasets and experiments are written inside):  
   [Run in Google Colab](https://colab.research.google.com/drive/1VGMA1cHQ2ClTuXvKgVDcqhYA14YRgEX2?usp=sharing)
   
2. **Local Machine**: You can also test the framework on your local machine by following the setup instructions provided in the repository.




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


