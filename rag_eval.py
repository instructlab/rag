import asyncio
import json
from pathlib import Path

import numpy as np
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


def score(answer: str, model_1="System A"):
    if answer not in ["System A", "System B", "EQUAL"]:
        return float('NaN')
    if answer == model_1:
        return 1.0
    elif answer == "EQUAL":
        return 0.5
    else:
        return 0.0


@app.command()
def evaluate_responses(
    input_file: Path = typer.Argument(
        "rag_with_responses.jsonl", help="Input file with RAG responses", exists=True
    ),
    output_file: Path = typer.Option(
        "rag_with_responses_and_scores.jsonl", help="Output file with scores"
    ),
    model_0_key: str = typer.Option(
        "agent_rag_granite", help="Key for the responses of the model that is preferred when the mean score is closer to 0 in the input file"
    ),
    model_1_key: str = typer.Option(
        "agent_rag_openai", help="Key for the responses of the model that is preferred when the mean score is closer to 1 in the input file"
    ),
):
    print("\033[92mStarting evaluation of responses.\033[0m")
    print(f"\033[92mModel 0: {model_0_key}\033[0m")
    print(f"\033[92mModel 1: {model_1_key}\033[0m")
    # Read the JSONL file
    with open(input_file, "r", encoding="utf-8") as file:
        questions = [json.loads(line) for line in file]

    # Update dictionary keys to match what is in the scoring prompts
    for question in questions:
        question["agent_rag_model_0"] = question.pop(model_0_key)
        question["agent_rag_model_1"] = question.pop(model_1_key)

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

        score_a = score(answer_a, model_1="System B")
        score_b = score(answer_b, model_1="System A")

        avg_score = (score_a + score_b) / 2

        questions[i]["judgement_when_model_0_is_system_A"] = result_a.content
        questions[i]["judgement_when_model_0_is_system_B"] = result_b.content
        questions[i]["avg_gpt4_based_judgement"] = avg_score
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as file:
        for item in questions:
            json.dump(item, file)
            file.write("\n")
    
    scores = np.array([q["avg_gpt4_based_judgement"] for q in questions])
    mean_score = np.nanmean(scores) if scores.size else 0
    nan_count = np.isnan(scores).sum()
    if nan_count > 0:
        print(f"\033[91mWarning: There were {nan_count} errors in the scoring.\033[0m")
    print("\033[96mgpt4 as a judge scores (average between positions):\033[0m", scores)
    print("\033[93mExplanation of scores:\033[0m")
    print("\033[93m- Score 1: Both evaluations favored Model 1.\033[0m")
    print("\033[93m- Score 0.75: One evaluation favored Model 1, the other found both models equal.\033[0m")
    print("\033[93m- Score 0.5: Evaluations found models equal or disagreed.\033[0m")
    print("\033[93m- Score 0.25: One evaluation favored Model 0, the other found both models equal.\033[0m")
    print("\033[93m- Score 0: Both evaluations favored Model 0.\033[0m")
    print("\033[93mNote: Evaluations are done twice to account for positional bias.\033[0m")
    print("\033[96mMean of gpt4 judge scores:\033[0m", mean_score)
    print("\033[93m- Mean score closer to 0: Preference for Model 0.\033[0m")
    print("\033[93m- Mean score closer to 1: Preference for Model 1.\033[0m")
    print(f"\033[92mJudgements and scores written to {output_file}.\033[0m")
    return scores


if __name__ == "__main__":
    app()
