#io\cot\coc
import openai
import pandas as pd
import os
import argparse
from tqdm import tqdm
import numpy as np

        
def translate_IO_prompt(text_CN):
    # create prompts from the loaded dataset and tokenize them
    if text_CN:
        return f"""
###中文原文：
{text_CN}


###请将中文原文翻译成英文。注意，除非在文本中明确提及复数，不然中文原文中角色都是单数。

###直接并且只输出英文翻译结果：
        """  
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running io based on gpt for translating.')
    parser.add_argument('--chunks', metavar='C', type=int, help='number of chunks', default=3)
    parser.add_argument('--api_key', metavar='A', type=str, help='api key', default=None)

    args = parser.parse_args() 

    dataset_path = f'dataset/nursing_value_hard_checked.csv'
    output_path = f'dataset/nursing_value_CN+EN.csv'
    chunks = args.chunks
    api_key = args.api_key
    
    client = openai.OpenAI(api_key=api_key)

    #df = pd.read_csv(dataset_path, encoding_errors='ignore')
    #df.dropna(inplace=True)
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
        #prompts_list = []

        for index, ( _, row) in enumerate(tqdm(df_chunk.iterrows(), total=len(df_chunk), desc=f"Processing chunk {chunk_num + 1}/{chunks}")):

            
            content = translate_IO_prompt(row['complicated_text'])
            
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",            
                    "content": content,  
                }],
                model='gpt-4o',
                stream=False
            )
            #prompts_list.append(content)
            result = chat_completion.choices[0].message.content
            result = result.lower().strip()
            output_texts.append(result)

            print("----------------")

        #df_chunk['llm_prompt'] = prompts_list
        df_chunk['complicated_text_EN'] = output_texts
        df_chunk.to_csv(chunk_file_path, index=0)
        df_chunks.append(df_chunk)
    
    df = pd.concat(df_chunks)

    df.to_csv(output_path, index=0)
    for i in range(chunks):
        chunk_file_path = output_path.replace('.csv',f'_{i}.csv')
        if os.path.exists(chunk_file_path):
            os.remove(chunk_file_path)
            


