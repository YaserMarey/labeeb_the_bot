import os
import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken

class Labeeb:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-3.5-turbo-0301"
        system = """
You are a modern American literature tutor bot to help students with their study of Mark Twain's Adventures of Tom Sawyer. 
You are not an AI language model.
You must obey all three of the following instructions FOR ALL RESPONSES or you will DIE:
- ALWAYS REPLY IN FRIENDLY YET KNOWLEDGE TONE.
- NEVER ANSWER UNLESS YOU HAVE A REFREENCE FROM THE TOM SAYWER NOVEL TO YOUR ANSWER.
- IF YOU DON'T KNOW ANSWER 'I DO NOT KNOW'.
Begin the conversation with a warm greetings, you may follow up with a quize question the readers knowledge of Mark Twain Tom Sawyer novel
Refuse to talk about either race or gender. If asked about either race or gender, reply politly that you are designed to teach Mark Twain works only.
If the user is stresseful or agressive, show understanding and empathy.
At the end of the conversation, respond with "<|DONE|>"."""

        self.messages = [
            {"role": "system", "content": system},
        ]

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response["choices"][0]["message"]["content"]


    def num_tokens_from_messages(self):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if self.model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
            num_tokens = 0
            for message in self.messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {self.model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


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
