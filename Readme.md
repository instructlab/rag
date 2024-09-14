# **How to Use This**

1. Create and activate a new Conda environment.
2. Install required Python packages with specified versions.
3. Set up environment variables (API keys and configurations).
4. Install and configure `ngrok` (optional).
5. Start the vLLM OpenAI-compatible API server.
6. Start the FastAPI server (`hybrid_pipeline2.py`).
7. Start the Streamlit application.
8. Use `ngrok` to expose local servers to the internet.
9. Configure applications to communicate with each other.
10. Verify the setup.

---

## **Step-by-Step Instructions**

### **1. Create and Activate the Conda Environment**

```bash
# Create a new environment named 'rag-test' with Python 3.11
mamba create -n rag-test python=3.11 -y

# Activate the environment
mamba activate rag-test
```

### **2. Install Required Python Packages with Specified Versions**

Install the core packages, ensuring versions match your `pip freeze` output.

```bash
git clone <this repo>
# Upgrade pip
cd <this repo>
pip install torch==2.3.0 #need to install before the other requirements
pip install -r requirements.txt
```

### **3. Set Environment Variables**

Set up necessary environment variables for API keys and configurations. Replace the placeholder values with your actual API keys.

```bash
# Set OpenAI API Key
export OPENAI_API_KEY='your_openai_api_key'

# Set Anthropic API Key
export ANTHROPIC_API_KEY='your_anthropic_api_key'

# Set the path to the model you're planning on using
export MODEL_PATH='/path/to/model' #/new_data/experiments/ss-bnp-p10/hf_format/samples_2795520

# set the path to the PDF you want to put in the RAG database.
export PDF_PATH='./2q24-cfsu-1.pdf'

# Optional: Set number of threads for OpenBLAS and OpenMP (adjust based on your CPU cores)
export OPENBLAS_NUM_THREADS=72
export OMP_NUM_THREADS=72
```

optionally you can put them in a `.env` file and we will load them in the code.

```.env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
MODEL_PATH=/path/to/model
PDF_PATH=./2q24-cfsu-1.pdf
OPENBLAS_NUM_THREADS=72
OMP_NUM_THREADS=72
```

### **4. Install and Configure ngrok (optional)**

This is only needed to expose local servers to the internet. If you're running everything locally, you can skip this step.

#### **Download and Install ngrok**

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

#### **Authenticate ngrok**

Sign up for an ngrok account to get your authtoken. Then configure ngrok with your authtoken:

```bash
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
```

Replace `YOUR_NGROK_AUTHTOKEN` with your actual ngrok authtoken.

### **5. Start the FastAPI Server**

```bash
cd <path_to_repo>
# Run the FastAPI application
CUDA_VISIBLE_DEVICES=7 python hybrid_pipeline2.py
```

Note: Make sure to use different CUDA_VISIBLE_DEVICES for the FastAPI server and the vLLM server to avoid GPU conflicts. For example, if you're using GPU 7 for the FastAPI server, use GPUs 0-3 for vLLM as shown in step 6.


### **6. Start the vLLM OpenAI-Compatible API Server**

Launch the vLLM server with your desired model. Replace the model path with the one you intend to use.

Remember to leave the first GPU to run the RAG model.

```bash
# Ray for VLLM to work
ray disable-usage-stats
ray start --head --num-cpus=32 --num-gpus=4 --disable-usage-stats
# Start vLLM server (adjust CUDA devices and model as needed)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model /new_data/experiments/ss-bnp-p10/hf_format/samples_2795520 \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9
```

- **Important:** Wait until vLLM is fully serving the model before proceeding to the next steps.
- Adjust `CUDA_VISIBLE_DEVICES`, `--tensor-parallel-size`, and other parameters based on your specific hardware configuration.
- The `--model` path should point to the directory containing the model files.
- Ensure that vLLM is serving the model on localhost:8000, as the FastAPI server expects this configuration.


### **7. Start the Streamlit Application**

Run your Streamlit application.

```bash
# Run the Streamlit app
cd <path_to_repo>
streamlit run streamlit_front.py
```

By default, Streamlit runs on port `8501`.

### **8. Use ngrok to Expose Local Servers**

#### **Expose Streamlit Application**

In a new terminal window:

```bash
ngrok http 8501
```

Again, note the generated ngrok URL for your Streamlit app.

### **10. Verify the Setup**

go to the ngrok URL or localhost:8501 to see the Streamlit app. If ngrok, you need to press `visit site` in blue to see the app.

use this question to test the pipeline:
```
What was BNP Paribas Group's net income attributable to equity holders for the first half of 2024, and how does it compare to the same period in 2023?
```