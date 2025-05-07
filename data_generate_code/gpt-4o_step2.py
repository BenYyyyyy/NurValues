#io\cot\coc
import openai
import pandas as pd
import os
import argparse
from tqdm import tqdm
import numpy as np

        
def generate_IO_prompt_align(case, nursing_value):
    # create prompts from the loaded dataset and tokenize them
    if nursing_value and case:
        return f"""
        ###现有的案例：
        {case}

        在上述现有的案例中，护士的行为遵守了{nursing_value}护理价值观。

        我希望你能够对上述的案例进行简单的反转，使其在保持原有的基础的场景的情况下，其中护士角色的行为违反{nursing_value}护理价值观。

        ###要求：
        1.请保持翻转后案例的文本量与原始案例的文本量一致。
        2.不要引入额外的细节，不要捏造原文没有提到的故事
        3.请仅输出翻转后的案例，不要附加任何额外的解释。

        ###输出：

        """  
    
def generate_IO_prompt_violate(case, nursing_value):
    # create prompts from the loaded dataset and tokenize them
    if nursing_value and case:
        return f"""
        ###现有的案例：
        {case}

        在上述现有的案例中，护士的行为违反了{nursing_value}护理价值观。

        我希望你能够对上述的案例进行简单的反转，使其在保持原有的基础的场景的情况下，其中护士角色的行为遵守{nursing_value}护理价值观。

        ###要求：
        1.请保持翻转后案例的文本量与原始案例的文本量一致。
        2.不要引入额外的细节，不要捏造原文没有提到的故事
        3.请仅输出翻转后的案例，不要附加任何额外的解释。

        ###输出：

        """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running io based on gpt for flipping case.')
    parser.add_argument('--chunks', metavar='C', type=int, help='number of chunks', default=3)
    parser.add_argument('--api_key', metavar='A', type=str, help='api key', default=None)

    args = parser.parse_args() 

    dataset_path = f'dataset/nursing_value_original_annotated.csv'
    output_path = f'dataset/nursing_value_easy_raw.csv'
    chunks = args.chunks
    api_key = args.api_key
    
    client = openai.OpenAI(api_key=api_key)

    df = pd.read_csv(dataset_path)

    chunk_size = int(np.ceil(len(df) / chunks))
    df_chunks = []

    for chunk_num in range(chunks):
        chunk_file_path = output_path.replace('.csv',f'_{chunk_num}.csv')

        if os.path.exists(chunk_file_path):
            df_chunk = pd.read_csv(chunk_file_path)
            df_chunks.append(df_chunk)
            continue

        df_chunk = df[chunk_num*chunk_size:min(len(df), (chunk_num+1)*chunk_size)]
        output_texts = []
        prompts_list = []

        for index, ( _, row) in enumerate(tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Processing chunk {chunk_num + 1}/{chunks}")):
            if pd.notna(row['Nursing_Value_2']):  
                nursing_value = f"{row['Nursing_Value_1']},{row['Nursing_Value_2']}"
            else:
                nursing_value = f"{row['Nursing_Value_1']}"
            
            if row['Alignment'] == 1:
                content = generate_IO_prompt_align(row['original_text'], nursing_value)
            else:
                content = generate_IO_prompt_violate(row['original_text'], nursing_value)
            
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",            
                    "content": content,  
                }],
                model='gpt-4o',
                stream=False
            )
            prompts_list.append(content)
            result = chat_completion.choices[0].message.content
            result = result.lower().strip()
            output_texts.append(result)

            print("----------------")

        df_chunk['llm_prompt'] = prompts_list
        df_chunk['flipped_text'] = output_texts
        df_chunk.to_csv(chunk_file_path, index=0)
        df_chunks.append(df_chunk)
    
    df = pd.concat(df_chunks)

    df.to_csv(output_path, index=0)
    for i in range(chunks):
        chunk_file_path = output_path.replace('.csv',f'_{i}.csv')
        if os.path.exists(chunk_file_path):
            os.remove(chunk_file_path)
            


