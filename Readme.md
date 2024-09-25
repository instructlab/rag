# **RAG-based Question Answering System**
## **Introduction**
This project implements a Retrieval-Augmented Generation (RAG) based question answering system. It combines the power of large language models with a document retrieval system to provide accurate and context-aware answers to user queries.
Key features of this system include:
1. Multiple LLM Support: The system can use different language models including OpenAI's GPT models, Anthropic's Claude, and a custom "Granite" model served via vLLM.
2. Vector Database: Utilizes Milvus as a vector database for efficient storage and retrieval of document embeddings.
3. Hybrid Search: Implements a hybrid search approach combining dense and sparse embeddings for improved retrieval accuracy.
Streamlit Frontend: Provides a user-friendly web interface for interacting with the system.
Evaluation Tools: Includes scripts for generating responses to a set of questions and evaluating the quality of responses against GPT-4.
6. Customizable: Allows for easy configuration of different components including the choice of embedding models, language models, and retrieval parameters.
This README will guide you through the setup process, explain how to use the various components of the system, and provide instructions for running evaluations.

## **Prerequisites**
- Python 3.11 or higher
- Conda or Mamba (for environment management)
- CUDA-compatible GPU(s) for running vLLM and the RAG server
- Access to OpenAI API (for GPT-4 evaluations)
- Access to Anthropic API (if using Claude model)
- (Optional) ngrok for exposing local servers to the internet

## **Installation**

### **1. Create and Activate the Conda Environment**

Create a new environment named 'rag-test' with Python 3.11
```bash
conda create -n rag-test python=3.11 -y
```
Activate the environment
```bash
conda activate rag-test
```


### **2. Install Required Python Packages**

Clone the repository and install the required packages:

```bash
git clone <repository_url>
cd <repository_directory>
```
Install PyTorch separately (adjust CUDA version as needed)
```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```
Install other requirements
```bash
pip install -r requirements.txt
```


### **3. Set Environment Variables**

Set up necessary environment variables for API keys and configurations. Create a `.env` file in the project root directory with the following content:

```bash
# Set OpenAI API Key
export OPENAI_API_KEY='your_openai_api_key'

# Set Anthropic API Key
export ANTHROPIC_API_KEY='your_anthropic_api_key'

export OPENBLAS_NUM_THREADS=72
export OMP_NUM_THREADS=72
```

Setting `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` limits the number of threads used by OpenBLAS and OMP, respectively. Not setting these can lead to "resource temporarily unavailable" errors, especially in containerized environments.


### **4. Start the Ray Server**

```bash
ray stop # Stop any existing Ray instances
CUDA_VISIBLE_DEVICES=4,5,6,7 ray start --head --num-cpus=32 --num-gpus=4 --disable-usage-stats
```

When running the RAG server and other GPU-intensive tasks, it's crucial to manage GPU resources effectively to avoid out-of-memory (OOM) errors. Ensure that you use different GPUs for the RAG server and other tasks to prevent resource contention and potential OOM issues.

For example, if you are using GPUs 4, 5, 6, and 7 for the Ray server, you should use different GPUs for the RAG server. You can specify the GPUs for the RAG server using the `CUDA_VISIBLE_DEVICES` environment variable to allocate different GPUs and mitigate the risk of OOM errors.


### **5. Start the vLLM Server**

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
--model /new_data/experiments/ss-bnp-p10/hf_format/samples_2795520 \
--dtype float16 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.8
```
Adjust the `--model` path and GPU settings as needed for your setup.

vLLM uses the Ray server, which in turn utilizes OPENBLAS and OMP threads for efficient parallel processing.

### **6. Start the RAG Server**

```bash
CUDA_VISIBLE_DEVICES=0 python rag_server.py run-server \
--reload-docs \
--milvus-db-path "milvus_demo.db" \
--collection-name "hybrid_pipeline" \
--file-paths "2q24-cfsu-1.pdf"
```

**Note:** The `--reload-docs` flag should be used only when the PDFs are updated. If this flag is set, the server will reload and re-index the documents, which can be time-consuming. For more details, refer to the `create_index` function in `milvus_index.py` and the `run_server` function in `rag_server.py`.

The RAG (Retrieval-Augmented Generation) server is designed to handle complex queries by leveraging a combination of retrieval and generation techniques. It uses a Milvus vector store to index and retrieve relevant documents and integrates with various language models to generate responses based on the retrieved information.

The `rag_server.py` file sets up the FastAPI server and defines endpoints for querying the RAG system. It initializes the necessary components, such as the reranker and the index, and provides functions to create query engines and handle incoming queries.

The `milvus_index.py` file contains the logic for creating and managing the Milvus vector store index. It defines how documents are loaded, parsed, and embedded into the vector store, enabling efficient retrieval of relevant information during query processing using hybrid search. The hybrid search combines dense and sparse embeddings for improved accuracy. Key parameters such as `milvus_db_path`, `collection_name`, and `reload_docs` are defined in `milvus_index.py` and utilized in `rag_server.py` to set up the RAG server.


### **7. Start the Streamlit Frontend**

```bash
streamlit run streamlit_front.py
```

The Streamlit frontend provides an interactive interface for querying different language models (LLMs) such as Claude, OpenAI, and Granite. Users can select one or more LLMs and input their query through a chat interface. The frontend sends the query to the backend server, which processes it and returns the responses from the selected LLMs. The responses are then displayed in the chat interface for the user to review. The backend server handles retries and error management to ensure reliable communication with the LLMs.


### **8. (Optional) Set Up ngrok for External Access**

If you need to expose your local servers to the internet:

1. Install ngrok:
Visit the [ngrok download page](https://ngrok.com/download) and download the appropriate version for your operating system.

For Linux:

```bash
# Download ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz

# Extract the archive
tar -xvzf ngrok-v3-stable-linux-amd64.tgz

# Move ngrok to a directory in your PATH
mv ngrok /usr/local/bin/
```

2. Configure ngrok:
Sign up for an ngrok account to get your authtoken. Then configure ngrok with your authtoken:

```bash
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
```

Replace `YOUR_NGROK_AUTHTOKEN` with your actual ngrok authtoken.

3. Expose the Streamlit app:
```bash
ngrok http 8501
```

Now your RAG-based Question Answering System should be set up and running!

### **9. Verify the Setup**

go to the ngrok URL or localhost:8501 to see the Streamlit app. If ngrok, you need to press `visit site` in blue to see the app.

use this question to test the pipeline:
```
What was BNP Paribas Group's net income attributable to equity holders for the first half of 2024, and how does it compare to the same period in 2023?
```

## Evaluation
The evaluation process involves generating responses to a set of questions using two different RAG models (Granite and OpenAI) and evaluating the quality of these responses using GPT-4. The evaluation script `rag_eval.py` provides two main commands: `generate_responses` and `evaluate_responses`. These commands allow you to generate responses to a set of questions and evaluate the responses based on GPT-4's judgments.


### CLI Arguments

The `rag_eval.py` script uses Typer to create a command-line interface with two main commands:

#### a) `generate_responses`:
- `dataset_path`: Path to the input JSONL file (default: `"rag_questions.jsonl"`)
- `output_path`: Path for the output JSONL file with responses (default: `"rag_with_responses.jsonl"`)
- `api_url`: URL for the API to fetch responses (default: `"http://localhost:8001"`)

#### b) `evaluate_responses`:
- `input_file`: Path to the JSONL file with RAG responses (default: `"rag_with_responses.jsonl"`)
- `output_file`: Path for the output JSONL file with scores (default: `"rag_with_responses_and_scores.jsonl"`)
- `model_1_key`: Key for model 1 responses in the input file (default: `"agent_rag_granite"`)
- `model_2_key`: Key for model 2 responses in the input file (default: `"agent_rag_openai"`)

### Dataset Format Assumptions

The script assumes the following format for the input JSONL files:

#### a) Input dataset (`rag_questions.jsonl`):
Each line is a JSON object containing:
- `document`: The context document
- `question`: The question to be answered
- `Ground Truth`: The ground truth answer


### Evaluation Process

The evaluation process involves the following steps:


#### dataset format

```json
{
  "document": "The XYZ Corporation, founded in 1985, specializes in manufacturing widgets. Their production process involves three main stages: sourcing raw materials, assembly, and quality control. The company has a strong focus on sustainability, implementing eco-friendly practices throughout their supply chain. XYZ Corporation has a global presence with factories in Asia, Europe, and North America, employing over 10,000 people worldwide.",
  "question": "What are the main stages of XYZ Corporation's production process?",
  "Ground Truth": "The main stages of XYZ Corporation's production process are sourcing raw materials, assembly, and quality control.",
}
```

#### a) Generate responses:
- Reads the input JSONL file
- Sends questions to the API to get responses from both models
- Saves the responses in a new JSONL file

```bash
python rag_eval.py generate-responses --dataset-path "rag_questions.jsonl" --output-path "rag_with_responses.jsonl" --api-url "http://localhost:8001"
```


#### b) Evaluate responses:
- Reads the JSONL file with responses
- Uses GPT-4 to evaluate the responses changing the order of the answers (`SCORING_PROMPT_A` and `SCORING_PROMPT_B`)
- The prompts compare the responses from both models based on various criteria and uses GPT-4 as judge
- Calculates an average score based on the two evaluations
- Saves the evaluations and scores in the final output JSONL file

```bash
python rag_eval.py evaluate-responses --input-file "rag_with_responses.jsonl" --output-file "rag_with_responses_and_scores.jsonl" --model-1-key "agent_rag_granite" --model-2-key "agent_rag_openai"
```