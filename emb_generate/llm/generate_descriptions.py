import json
import time
from tqdm import tqdm
from openai import OpenAI
import config

def load_items(path):
    with open(path, "r", encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_descriptions(item_name, client):
    prompt = f"""
    您是一位牙科專家。請針對以下牙科處置項目「{item_name}」，產出 {config.N_DESCRIPTIONS} 則準確且多樣化的描述。
    每則描述需符合以下條件：
    - 事實正確
    - 簡短（不超過一句話）
    - 回應內容使用繁體中文
    - 各描述之間請以 <SEP> 分隔
    - 只要給描述即可，不需要重複其名稱也不要加上編號
    """
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional dental expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        content = response.choices[0].message.content
        descriptions = [desc.strip() for desc in content.split("<SEP>") if desc.strip()]
        return descriptions[:config.N_DESCRIPTIONS]
    except Exception as e:
        print(f"Error for '{item_name}':", e)
        return []

def main():
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    items = load_items(config.ITEM_LIST_PATH)
    results = {}
    for item in tqdm(items, desc="Generating descriptions"):
        descs = get_descriptions(item, client)
        if descs:
            results[item] = descs
        break

    with open(config.DESCRIPTION_OUTPUT_PATH, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved descriptions to {config.DESCRIPTION_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
