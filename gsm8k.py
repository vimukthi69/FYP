from openai import OpenAI
import pandas as pd
import os
import re

client = OpenAI(api_key='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')

system_prompt = """
You will be provided with a math question. What you have to do is, perform reasoning under steps given below to achieve 
the correct answer.

<goal_detector>detect what you have to achieve..</goal_detector>
<plan_generator>make a plan on how you are going to achieve the detected goal..</plan_generator>
<projector>project what will happen if you execute the generated plan..</projector>
<executer>final answer comes here as you planned above..</executer>

If the projector projects generated plan is not achieving the correct answer, execute 'plan_generator' step until 
projector projects generated plan is going to achieve the correct answer.

Make sure to include all the 8 opening and closing tags in your final answer.
"""


def generate_answer_from_llm(prompt, user_input):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {
                "role": "user",
                "content": f"{user_input}"
            }
        ]
    )

    return completion.choices[0].message.content


def append_to_csv(file_path, question, reasoning, answer):
    qna_pair = pd.DataFrame([[question, reasoning, answer]], columns=['instruction', 'chosen_response', 'rejected_response'])
    qna_pair.to_csv(file_path, mode='a', header=False, index=False)


def slice_answer(llm_output):
    match = re.search(r'<executer>(.*?)</executer>', llm_output, re.DOTALL)
    return match.group(1) if match else None


def generate_dataset(file_path, output_file_path, start_index=14830):
    df = pd.read_csv(file_path)
    for index, row in df.iloc[start_index:].iterrows():
        print("Executing question : ", index)
        question = row['question']

        print("Executing question : ", question)

        generated_answer = generate_answer_from_llm(system_prompt, question)
        print("Answer with reasoning steps : ", generated_answer)
        sliced_answer = row['answer']
        append_to_csv(output_file_path, question, generated_answer, sliced_answer)


input_dataset_path = 'C:\\Users\\vimuk\\PycharmProjects\\dataset_gen\\dataset_original\\dolr1.csv'
output_dataset_path = 'C:\\Users\\vimuk\\PycharmProjects\\dataset_gen\\generated_dataset_reward\\pair_gsm8k_v2.csv'
generate_dataset(input_dataset_path, output_dataset_path, start_index=14830)

