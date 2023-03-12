import os
import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken

class Labeeb:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.messages = [
            {"role": "system", "content": "You are a moderen american literature tutor bot to help students with their study of Mark Twain's Advantures of Tom Sawyer. You answer as Answer the question as truthfully as possible, and if you're unsure of the answer, say Sorry, I don't know "},
        ]

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]["content"]

    def construct_prompt(self, question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
        most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = df.loc[section_index]
            chosen_sections_len += document_section.tokens + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break
            chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))
                
        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))
        
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
        
        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    def execute(self):
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=self.messages, temperature=self.temperature
            )
            self.token_total += completion["usage"]["total_tokens"]
            return completion["choices"][0]["message"]["content"]
