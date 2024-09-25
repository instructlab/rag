import asyncio
import json
from pathlib import Path

import typer
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


from prompt_templates import SCORING_PROMPT_A, SCORING_PROMPT_B


app = typer.Typer()


async def fetch_response(client, llm, query_str, api_url="http://localhost:8001"):
    response = await client.post(
        f"{api_url}/get_response", 
        json={"llm": llm, "query_str": query_str},
        timeout=600,
    )
    response_json = response.json()
    print(
        f"\033[92mResponse from {llm} for question:\033[0m \033[95m'{query_str}'\033[0m\n\033[0m{response_json['response']}\033[0m"
    )
    return response_json


@app.command()
def generate_responses(
    dataset_path: Path = typer.Argument(
        Path("rag_questions.jsonl"), help="Evaluation Dataset Path", exists=True
    ),
    output_path: Path = typer.Option(
        "rag_with_responses.jsonl", help="Output Dataset Path"
    ),
    api_url: str = typer.Option(
        "http://localhost:8001", help="API URL for fetching responses"
    ),
):
    # Read the JSONL file
    with open(dataset_path, "r", encoding="utf-8") as file:
        questions = [json.loads(line) for line in file]

    async def process_questions():
        async with httpx.AsyncClient() as client:
            tasks = []
            for item in questions:
                tasks.append(
                    fetch_response(client, "granite", item["question"], api_url=api_url)
                )
                tasks.append(
                    fetch_response(client, "openai", item["question"], api_url=api_url)
                )
            return await asyncio.gather(*tasks)

    # Run the async function
    results = asyncio.run(process_questions())

    # Process responses and update items
    for i, item in enumerate(questions):
        item["agent_rag_granite"] = results[i * 2]["response"]
        item["agent_rag_openai"] = results[i * 2 + 1]["response"]

    # Save results to the output file
    with open(output_path, "w", encoding="utf-8") as file:
        for item in questions:
            json.dump(item, file)
            file.write("\n")

    print(f"Results saved to {output_path}")


@app.command()
def evaluate_responses(
    input_file: Path = typer.Argument(
        "rag_with_responses.jsonl", help="Input file with RAG responses", exists=True
    ),
    output_file: Path = typer.Option(
        "rag_with_responses_and_scores.jsonl", help="Output file with scores"
    ),
    model_1_key: str = typer.Option(
        "agent_rag_granite", help="Key for model 1 responses in the input file"
    ),
    model_2_key: str = typer.Option(
        "agent_rag_openai", help="Key for model 2 responses in the input file"
    ),
):
    print("\033[92mStarting evaluation of responses.\033[0m")
    print(f"\033[92mModel 1: {model_1_key}\033[0m")
    print(f"\033[92mModel 2: {model_2_key}\033[0m")
    # Read the JSONL file
    with open(input_file, "r", encoding="utf-8") as file:
        questions = [json.loads(line) for line in file]

    # Update dictionary keys to match what is in the scoring prompts
    for question in questions:
        question["agent_rag_model_1"] = question.pop(model_1_key)
        question["agent_rag_model_2"] = question.pop(model_2_key)

    # Create the ChatOpenAI model
    model = ChatOpenAI(
        model="gpt-4",
        temperature=0.0,
        timeout=1200,
    )

    # Create the prompt templates
    prompt_a = ChatPromptTemplate.from_template(SCORING_PROMPT_A) # model A is granite
    prompt_b = ChatPromptTemplate.from_template(SCORING_PROMPT_B) # model B is granite

    # Create the chains
    chain_a = prompt_a | model
    chain_b = prompt_b | model

    # Define the async function to run tasks
    async def run_tasks():
        tasks = [chain_a.abatch(questions), chain_b.abatch(questions)]
        return await asyncio.gather(*tasks)

    # Run evaluations using asyncio.run
    results_a, results_b = asyncio.run(run_tasks())

    # Process results
    for i, (result_a, result_b) in enumerate(zip(results_a, results_b)):
        answer_a = result_a.content.split("<answer>")[-1].split("</answer>")[0].strip()
        answer_b = result_b.content.split("<answer>")[-1].split("</answer>")[0].strip()

        score_a = 1 if answer_a == "System A" else 0
        score_b = 1 if answer_b == "System B" else 0

        avg_score = (score_a + score_b) / 2

        questions[i]["judgement_when_model_1_is_first"] = result_a.content
        questions[i]["judgement_when_model_1_is_second"] = result_b.content
        questions[i]["avg_gpt4_based_judgement"] = avg_score

    # Save results
    with open(output_file, "w", encoding="utf-8") as file:
        for item in questions:
            json.dump(item, file)
            file.write("\n")
    
    scores = [q["avg_gpt4_based_judgement"] for q in questions]
    print("\033[96mgpt4 as a judge scores (average between positions):\033[0m", scores)
    mean_score = sum(scores) / len(scores) if scores else 0
    print("\033[96mMean of gpt4 judge scores:\033[0m", mean_score)
    print("\033[93mExplanation of scores:\033[0m")
    print("\033[93m- A score of 1 means both evaluations favored Model 2.\033[0m")
    print("\033[93m- A score of 0.5 means one evaluation favored Model 1 and the other favored Model 2.\033[0m")
    print("\033[93m- A score of 0 means both evaluations favored Model 1.\033[0m")
    print("\033[93mNote: The evaluations are done twice to account for positional bias, where the order of presented models might influence the judgment.\033[0m")
    print("\033[93m- A mean score closer to 0 indicates a preference for Model 1, while a mean score closer to 1 indicates a preference for Model 2.\033[0m")
    print(f"\033[92mJudgements and scores have been written {output_file}.\033[0m")
    return scores


if __name__ == "__main__":
    app()
