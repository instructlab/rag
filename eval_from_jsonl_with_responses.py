

import asyncio
import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from scoring_prompt import scoring_prompt_a, scoring_prompt_b
from langchain_core.prompts import ChatPromptTemplate


async def evaluate_rag_responses(input_file: Path):
    # Read the JSONL file
    with open(input_file, 'r') as file:
        questions = [json.loads(line) for line in file]

    # Create the ChatOpenAI model
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        timeout=1200,
    )

    # Create the prompt templates
    prompt_a = ChatPromptTemplate.from_template(scoring_prompt_a)
    prompt_b = ChatPromptTemplate.from_template(scoring_prompt_b)

    # Create the chain
    chain_a = prompt_a | model
    chain_b = prompt_b | model

    tasks = [
        chain_a.abatch(questions),
        chain_b.abatch(questions)
    ]
    results_a, results_b = await asyncio.gather(*tasks)

    # Parse the results and compute the average score
    for i, (result_a, result_b) in enumerate(zip(results_a, results_b)):
        # Extract the answer from each result
        answer_a = result_a.content.split('<answer>')[-1].split('</answer>')[0].strip()
        answer_b = result_b.content.split('<answer>')[-1].split('</answer>')[0].strip()

        # Convert answers to numerical scores
        score_a = 1 if answer_a == 'System B' else 0
        score_b = 1 if answer_b == 'System A' else 0

        # Compute the average score
        avg_score = (score_a + score_b) / 2

        # Add the average score to the corresponding question
        questions[i]['judgement_when_a_is_granite'] = result_b.content
        questions[i]['judgement_when_b_is_granite'] = result_a.content
        questions[i]['avg_gpt4_based_judgement'] = avg_score

    return questions

if __name__ == "__main__":
    results = asyncio.run(evaluate_rag_responses("/new_data/aldo/rag/rag_questions_with_responses.jsonl"))
    # Save results to the same JSONL file
    output_file = "/new_data/aldo/rag/rag_questions_with_responses_and_scores.jsonl"
    with open(output_file, 'w') as file:
        for item in results:
            json.dump(item, file)
            file.write('\n')
    scores = [r['avg_gpt4_based_judgement'] for r in results]
    print(scores)
    print(sum(scores)/len(scores))
    from IPython import embed; embed()