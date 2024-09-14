import streamlit as st
import aiohttp
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor


async def get_llm_response(llm, query):
    url = "http://localhost:8001/get_response"
    headers = {"Content-Type": "application/json"}
    data = {"llm": llm, "query_str": query}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                return llm, result["response"]
        except aiohttp.ClientError as e:
            return llm, f"Error: {str(e)}"


async def get_all_responses(llms, query):
    tasks = [get_llm_response(llm, query) for llm in llms]
    return await asyncio.gather(*tasks)


# Add a multiselect for LLM selection
llm_options = ["claude", "openai", "granite"]
selected_llms = st.multiselect("Select LLM(s):", llm_options)

if prompt := st.chat_input(
    "What was BNP Paribas Group's net income attributable to equity holders for the first half of 2024, and how does it compare to the same period in 2023?"
):
    st.chat_message("user").write(prompt)

    # Get responses for all selected LLMs concurrently
    responses = asyncio.run(get_all_responses(selected_llms, prompt))
    # Display responses
    for llm, response in responses:
        st.chat_message("assistant").write(f"{llm.capitalize()} response: {response}")
