import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from threading import Lock
import json_repair
import json 
from openai import OpenAI
import time
from requests.exceptions import RequestException
from core.config_utils import load_key

LOG_FOLDER = 'output/gpt_log'
LOCK = Lock()

def save_log(model, prompt, response, log_title = 'default', message = None):
    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_data = {
        "model": model,
        "prompt": prompt,
        "response": response,
        "message": message
    }
    log_file = os.path.join(LOG_FOLDER, f"{log_title}.json")
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_data)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
        
def check_ask_gpt_history(prompt, model, log_title):
    # check if the prompt has been asked before
    if not os.path.exists(LOG_FOLDER):
        return False
    file_path = os.path.join(LOG_FOLDER, f"{log_title}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item["prompt"] == prompt:
                    return item["response"]
    return False

def fix_base_url(base_url):
    # huoshan
    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        return base_url
    # general
    if 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    return base_url

def ask_gpt(prompt, response_json=True, valid_def=None, log_title='default'):
    api_set = load_key("api")
    llm_support_json = load_key("llm_support_json")
    with LOCK:
        history_response = check_ask_gpt_history(prompt, api_set["model"], log_title)
        if history_response:
            return history_response
    
    if not api_set["key"]:
        raise ValueError(f"⚠️API_KEY is missing")
    
    # 增加对JSON格式的明确要求 (使用英文提示，避免影响非中文任务)
    if response_json:
        prompt = prompt + "\nIMPORTANT: Ensure your response is strictly valid JSON format. All string values MUST be wrapped in quotes, especially the 'reflection' field values."
    
    messages = [{"role": "user", "content": prompt}]
    
    base_url = fix_base_url(api_set["base_url"])
    client = OpenAI(api_key=api_set["key"], base_url=base_url)
    response_format = {"type": "json_object"} if response_json and api_set["model"] in llm_support_json else None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion_args = {
                "model": api_set["model"],
                "messages": messages
            }
            if response_format is not None:
                completion_args["response_format"] = response_format
                
            response = client.chat.completions.create(**completion_args)
            
            if response_json:
                try:
                    content = response.choices[0].message.content
                    # 先尝试直接解析
                    try:
                        response_data = json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            # 使用 json_repair 进行修复
                            response_data = json_repair.loads(content)
                        except Exception as e:
                            # 手动处理常见问题
                            print(f"尝试手动修复 JSON: {str(e)}")
                            # 检查是否有未加引号的字段
                            fixed_content = content
                            
                            # 改进：增强reflection字段的处理
                            if '"reflection":' in fixed_content:
                                import re
                                # 处理未加引号的reflection值 - 更强大的正则表达式
                                # 这个正则表达式会捕获reflection后面到下一个引号前或逗号前的所有内容
                                fixed_content = re.sub(
                                    r'"reflection"\s*:\s*([^"][^,}]*?)(?=,|\s*})',
                                    r'"reflection": "\1"',
                                    fixed_content
                                )
                                
                                # 处理其他可能的格式问题
                                # 修复双引号问题
                                fixed_content = fixed_content.replace('""', '"')
                                # 修复缺少逗号的问题
                                fixed_content = re.sub(r'"\s*}\s*{', '"},\n{', fixed_content)
                                
                                print(f"修复后的JSON前100个字符: {fixed_content[:100]}...")
                            
                            try:
                                response_data = json.loads(fixed_content)
                            except json.JSONDecodeError:
                                try:
                                    response_data = json_repair.loads(fixed_content)
                                except Exception as e2:
                                    # 如果修复仍然失败，尝试截断或完善JSON
                                    try:
                                        print("尝试截断/完善JSON...")
                                        # 尝试修复常见错误：找到最后一个有效的JSON子串
                                        if content.strip().startswith('{') and '}' in content:
                                            last_brace_index = content.rindex('}')
                                            truncated_json = content[:last_brace_index+1]
                                            response_data = json_repair.loads(truncated_json)
                                            print("通过截断JSON修复成功")
                                        else:
                                            # 如果仍然失败，记录错误并继续重试
                                            response_data = content
                                            print(f"❎ json_repair parsing failed. Retrying: '''{content[:200]}...'''")
                                            save_log(api_set["model"], prompt, content, log_title="error", message=f"json_repair parsing failed: {str(e2)}")
                                            if attempt == max_retries - 1:
                                                raise Exception(f"JSON parsing still failed after {max_retries} attempts: {e2}\n Please check your network connection or API key or `output/gpt_log/error.json` to debug.")
                                            continue
                                    except Exception as e3:
                                        response_data = content
                                        print(f"❎ 所有JSON修复方法都失败了。Retrying: '''{content[:200]}...'''")
                                        save_log(api_set["model"], prompt, content, log_title="error", message=f"All json repairs failed: {str(e3)}")
                                        if attempt == max_retries - 1:
                                            raise Exception(f"JSON parsing still failed after {max_retries} attempts with all repair methods: {e3}")
                                        continue
                    
                    # check if the response is valid, otherwise save the log and raise error and retry
                    if valid_def:
                        valid_response = valid_def(response_data)
                        if valid_response['status'] != 'success':
                            save_log(api_set["model"], prompt, response_data, log_title="error", message=valid_response['message'])
                            raise ValueError(f"❎ API response error: {valid_response['message']}")
                    
                    break  # Successfully accessed and parsed, break the loop
                except Exception as e:
                    response_data = response.choices[0].message.content
                    print(f"❎ json_repair parsing failed. Retrying: '''{response_data[:200]}...'''")
                    save_log(api_set["model"], prompt, response_data, log_title="error", message=f"json_repair parsing failed.")
                    if attempt == max_retries - 1:
                        raise Exception(f"JSON parsing still failed after {max_retries} attempts: {e}\n Please check your network connection or API key or `output/gpt_log/error.json` to debug.")
            else:
                response_data = response.choices[0].message.content
                break  # Non-JSON format, break the loop directly
                
        except Exception as e:
            if attempt < max_retries - 1:
                if isinstance(e, RequestException):
                    print(f"Request error: {e}. Retrying ({attempt + 1}/{max_retries})...")
                else:
                    print(f"Unexpected error occurred: {e}\nRetrying...")
                time.sleep(2)
            else:
                raise Exception(f"Still failed after {max_retries} attempts: {e}")
    with LOCK:
        if log_title != 'None':
            save_log(api_set["model"], prompt, response_data, log_title=log_title)

    return response_data


if __name__ == '__main__':
    print(ask_gpt('hi there hey response in json format, just return 200.' , response_json=True, log_title=None))