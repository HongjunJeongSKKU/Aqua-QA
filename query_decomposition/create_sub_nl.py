import transformers
import torch
import os
import pandas as pd
import math
from tqdm import tqdm
import numpy as np
import click

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import set_seed


@click.command()
@click.option("--dataset", default="../data/CODEX")
@click.option("--gen_train", is_flag=True, default=False)
@click.option("--gen_valid", is_flag=True, default=False)
@click.option("--gen_test", is_flag=True, default=False)
@click.option("--llm_model", default="llama")
def main(dataset, gen_train, gen_valid, gen_test, llm_model):

    if llm_model == "llama":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto"
        )

    set_seed(0)

    if gen_train:
        mode = "train"
    elif gen_valid:
        mode = "valid"
    elif gen_test:
        mode = "test"
    else:
        print("Set the mode (--gen_train, --gen_valid, --gen_test)")
        exit(1)

    nl_df = pd.read_parquet(f"{dataset}/{mode}-drop_dup.parquet")
    nl_df_query = nl_df["query"].reset_index(drop=True)

    batch_size = 20
    result = []  # for generated text
    for j in tqdm(range(math.ceil(len(nl_df_query.dropna()) / batch_size))):

        batch_messages = []  # for batch sample
        for i in range(batch_size):  # batch size
            index = i + j * batch_size
            if index >= len(nl_df_query) or type(nl_df_query[index]) == type(np.nan):
                break

            messages1 = [
                {
                    "role": "system",
                    "content": "You are an intelligent assistant and a natural language question generator that creates effective and concise questions. ONLY RETURN THE SUB-QUESTION(QUESTIONS) WITHOUT ANY EXPLANATION.",
                },
                {
                    "role": "assistant",
                    "content": """Instructions:

- Extract every claim from the provided query.
- Resolve any coreference for clarity.
- Convert each claim into a concise (less than 15 words).
- Generate no more than 5 sub-questions.
- Generate sub-questions only based on information available in the original question.
- Separate multiple sub-questions with ' / ', except for certain 'p' type queries. A 'p' type query involves a sequence of steps or a path where one piece of information is used to find another (for example, "Which educational institutions have the same colors as the Indiana Pacers?" requires first finding the colors of the Indiana Pacers, then finding educational institutions with those colors). Do not separate these.
- DO NOT RETURN ANYTHING BUT THE ANSWER.
                        
Examples:

1. Query: Which football teams have a Forward, a Goalkeeper, and a Defender on their roster?"
Response: Which football teams have a Forward? / Which football teams have a Goalkeeper? / Which football teams have a Defender?

2. Query: Who are the University of Miami graduates that are married to Melanie Griffith?
Response: Who are the University of Miami graduates? / Who are married to Melanie Griffith?

3. Query: What is Billy Boyd's profession?
Response: What is Billy Boyd's profession?""",
                },
                {
                    "role": "user",
                    "content": f"""Complete the following:

Question: {nl_df_query[index]}
Response:""",
                },
            ]

            batch_messages.append(messages1)

        # 토큰화 및 텐서 변형
        model_inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # 입력 텐서의 크기를 계산
        input_length = model_inputs.shape[1]

        # 각 입력에 대한 응답을 생성
        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=192,
            min_length=3,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
        )

        # 생성된 토큰 디코딩
        output_texts = tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )
        for output in output_texts:
            print(output)
        result.extend(output_texts)
    nl_df["sub"] = result
    nl_df["sub"] = nl_df["sub"].apply(lambda x: x.split(" / "))
    nl_df.to_parquet(f"{dataset}/{mode}-subq_drop_dup.parquet")

    print(nl_df)


if __name__ == "__main__":
    main()
