import streamlit as st
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_random


def log_retry_attempt(retry_state):
    attempt = retry_state.attempt_number
    st.warning(f"Attempt {attempt} failed for {retry_state.args[0]}. Retrying...")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random(min=60, max=120),
    before_sleep=log_retry_attempt,
    retry_error_callback=lambda retry_state: (
        retry_state.args[0],
        "Error: All retry attempts failed",
    ),
)
async def get_llm_response(llm, query):
    url = "http://localhost:8001/get_response"
    headers = {"Content-Type": "application/json"}
    data = {"llm": llm, "query_str": query}

    async with httpx.AsyncClient() as client:
        try:
            post_response = await client.post(
                url, headers=headers, json=data, timeout=600
            )
            post_response.raise_for_status()
            result = post_response.json()
            return llm, result["response"]
        except httpx.TimeoutException:
            return llm, "Error: Request timed out after 10 minutes"
        except httpx.HTTPStatusError as e:
            return llm, f"Error: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return llm, f"Error: {str(e)}"


async def get_all_responses(llms, query):
    tasks = [get_llm_response(llm, query) for llm in llms]
    return await asyncio.gather(*tasks)


# Add a multiselect for LLM selection
llm_options = ["claude", "openai", "granite"]
selected_llms = st.multiselect("Select LLM(s):", llm_options)

if prompt := st.chat_input(
    "What was BNP Paribas Group's net income attributable to equity holders "
    "for the first half of 2024, and how does it compare to the same period in 2023?"
):
    st.chat_message("user").write(prompt)

    # Get responses for all selected LLMs concurrently
    responses = asyncio.run(get_all_responses(selected_llms, prompt))
    # Display responses
    for llm, response in responses:
        st.chat_message("assistant").write(f"{llm.capitalize()} response: {response}")
