"""
Entrypoint for Gradio, see https://gradio.app/
"""

import platform
import os
import json
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Tuple, Dict

import gradio as gr
import requests  # 使用 requests 进行 HTTP 请求

from screeninfo import get_monitors

# 定义 APIProvider 枚举类
class APIProvider(Enum):
    FRIDAY = "friday"

# 定义默认模型映射
PROVIDER_TO_DEFAULT_MODEL_NAME = {
    APIProvider.FRIDAY.value: "anthropic.claude-3.5-sonnet-v2"  # 更新模型名称
}

# 模拟的 ToolResult 类和 get_screen_details 函数，替换为实际实现
class ToolResult:
    def __init__(self, output=None, error=None, base64_image=None):
        self.output = output
        self.error = error
        self.base64_image = base64_image

def get_screen_details():
    # 模拟函数，替换为实际的屏幕详情获取
    screens = get_monitors()
    screen_options = [f"Screen {i}" for i, _ in enumerate(screens)]
    primary_index = 0 if screens else None
    return screen_options, primary_index

CONFIG_DIR = Path("~/.friday").expanduser()
API_ID_FILE = CONFIG_DIR / "app_id"

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"

# 定义消息发送者枚举类
class Sender(str, Enum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

# 加载和保存函数
def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"Debug: Error loading {filename}: {e}")
    return None

def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # 确保只有用户可以读写文件
        file_path.chmod(0o600)
    except Exception as e:
        print(f"Debug: Error saving {filename}: {e}")

# 调用 Friday API 并获取响应
def call_friday_api(app_id: str, model: str, system_prompt: str, messages: List[dict]) -> str:
    api_url = "https://aigc.sankuai.com/v1/claude/aws/v1/messages"  # 普通 API

    headers = {
        "Authorization": f"Bearer {app_id}",
        "Content-Type": "application/json"
    }

    payload = {
        "max_tokens": 100,  # 根据需要调整
        "stream": False,    # 如果需要流式调用，设置为 True
        "temperature": 1,
        "top_k": 250,
        "top_p": 0.999,
        "model": model,
        "system": system_prompt,
        "stop_sequences": [
            "Human:"
        ],
        "anthropic_beta": ["computer-use-2024-10-22"],  # 添加 anthropic_beta 参数
        "messages": [
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            }
            for msg in messages
        ]
    }

    # 打印请求 payload 以调试
    print("API Request Payload:", json.dumps(payload, indent=2, ensure_ascii=False))

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        print("API Response Status:", response.status_code)
        print("API Response Text:", response.text)
        if response.status_code == 200:
            response_data = response.json()
            # 假设 response_data["content"] 是一个包含文本块的列表
            content = response_data.get("content", [])
            bot_response = "".join([block.get("text", "") for block in content if block.get("type") == "text"])
            print("Bot Response:", bot_response)
            return bot_response
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception: {str(e)}"

# 设置默认模型
DEFAULT_MODEL = PROVIDER_TO_DEFAULT_MODEL_NAME.get(APIProvider.FRIDAY.value, "anthropic.claude-3.5-sonnet-v2")  # 使用更新后的默认模型

# 定义主要处理函数
def process_input(user_input: str, chat_history: List[Tuple[str, str]], settings: Dict) -> List[Tuple[str, str]]:
    print("Processing input:", user_input)
    # Extract settings
    app_id = settings.get("app_id", "")
    model = settings.get("model", DEFAULT_MODEL)
    system_prompt = settings.get("custom_system_prompt", "")
    print("Initial system_prompt:", system_prompt)
    
    # 如果系统提示为空，设置默认值
    if not system_prompt.strip():
        system_prompt = "You are a helpful assistant."
        print("Default system_prompt set:", system_prompt)
    
    if not app_id:
        print("AppId is missing.")
        return chat_history + [("System", "AppId is missing. Please set it in settings.")]
    
    # Append user input to chat history
    updated_history = chat_history + [(Sender.USER.value, user_input)]
    print("Updated chat history with user input:", updated_history)
    
    # Prepare API messages
    # Do not include 'role': 'system' in messages, use 'system' field instead
    api_messages = []
    
    # Add all prior user and assistant messages
    for role, message in updated_history:
        api_messages.append({"role": role, "content": message})
    
    print("API Messages:", api_messages)
    
    # Call Friday API
    bot_response = call_friday_api(app_id, model, system_prompt, api_messages)
    
    print("Received bot response:", bot_response)
    
    # Append bot response to chat history
    updated_history.append((Sender.BOT.value, bot_response))
    print("Updated chat history with bot response:", updated_history)
    
    return updated_history

# 初始化 Gradio 界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Gradio 状态用于设置
    settings_state = gr.State({
        "app_id": load_from_storage("app_id") or "",
        "model": DEFAULT_MODEL,
        "custom_system_prompt": "You are a helpful assistant.",  # 默认系统提示
        # 根据需要添加其他设置
    })

    # Gradio 状态用于聊天历史
    chat_history_state = gr.State([])

    # 布局
    gr.Markdown("# Computer Use OOTB")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(WARNING_TEXT)

    with gr.Accordion("Settings", open=False):
        with gr.Row():
            with gr.Column():
                model_input = gr.Textbox(label="Model", value=DEFAULT_MODEL)
            with gr.Column():
                app_id_input = gr.Textbox(
                    label="Friday AppId",
                    type="password",
                    value=settings_state.value.get("app_id", ""),
                    interactive=True,
                )
            with gr.Column():
                custom_prompt_input = gr.Textbox(
                    label="System Prompt Suffix",
                    value=settings_state.value.get("custom_system_prompt", "You are a helpful assistant."),
                    interactive=True,
                )
            with gr.Column():
                screen_options, primary_index = get_screen_details()
                screen_selector = gr.Dropdown(
                    label="Select Screen",
                    choices=screen_options,
                    value=screen_options[primary_index] if screen_options else None,
                    interactive=True,
                )
            with gr.Column():
                only_n_images_slider = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    value=2,
                    interactive=True,
                )
            # hide_images = gr.Checkbox(label="Hide screenshots", value=False)

        # 保存设置的回调函数
        def update_settings(model, app_id, custom_prompt, screen, only_n_images):
            settings = {
                "model": model,
                "app_id": app_id,
                "custom_system_prompt": custom_prompt,
                "selected_screen": screen,
                "only_n_most_recent_images": only_n_images,
                # "hide_images": hide_images,
            }
            # 保存 app_id 到存储
            save_to_storage("app_id", app_id)
            print("Settings updated:", settings)
            return settings

        # 绑定设置变化的回调
        model_input.change(update_settings, inputs=[model_input, app_id_input, custom_prompt_input, screen_selector, only_n_images_slider], outputs=settings_state)
        app_id_input.change(update_settings, inputs=[model_input, app_id_input, custom_prompt_input, screen_selector, only_n_images_slider], outputs=settings_state)
        custom_prompt_input.change(update_settings, inputs=[model_input, app_id_input, custom_prompt_input, screen_selector, only_n_images_slider], outputs=settings_state)
        screen_selector.change(update_settings, inputs=[model_input, app_id_input, custom_prompt_input, screen_selector, only_n_images_slider], outputs=settings_state)
        only_n_images_slider.change(update_settings, inputs=[model_input, app_id_input, custom_prompt_input, screen_selector, only_n_images_slider], outputs=settings_state)

    with gr.Accordion("Quick Start Prompt", open=False):
        with gr.Row():
            # 确保 ootb_examples.json 文件存在
            try:
                with open("examples/ootb_examples.json", "r", encoding="utf-8") as f:
                    merged_dict = json.load(f)
            except FileNotFoundError:
                merged_dict = {}
                print("Warning: 'examples/ootb_examples.json' not found.")

            # 定义下拉菜单
            initial_category = "Game Play"
            initial_second_options = list(merged_dict.get(initial_category, {}).keys()) if initial_category in merged_dict else []
            initial_third_options = list(merged_dict.get(initial_category, {}).get(initial_second_options[0], {}).keys()) if initial_second_options else []
            initial_text_value = merged_dict.get(initial_category, {}).get(initial_second_options[0], {}).get(initial_third_options[0], {}).get("prompt", "") if initial_second_options and initial_third_options else ""

            with gr.Column(scale=2):
                first_menu = gr.Dropdown(
                    choices=list(merged_dict.keys()), label="Task Category", interactive=True, value=initial_category if initial_category in merged_dict else None
                )

                second_menu = gr.Dropdown(
                    choices=initial_second_options, label="Software", interactive=True, value=initial_second_options[0] if initial_second_options else None
                )

                third_menu = gr.Dropdown(
                    choices=["Please select a task"] + initial_third_options if initial_third_options else ["Please select a task"], label="Task", interactive=True, value="Please select a task"
                )
            
            with gr.Column(scale=1):
                image_preview = gr.Image(label="Reference Initial State", height=260 - (318.75 - 280))
                hintbox = gr.Markdown("Task Hint: Selected options will appear here.")

            # 定义回调函数更新下拉菜单和文本框
            def update_second_menu(selected_category):
                second_options = list(merged_dict.get(selected_category, {}).keys())
                print("Updating second menu with:", second_options)
                return second_options

            def update_third_menu(selected_category, selected_option):
                third_options = list(merged_dict.get(selected_category, {}).get(selected_option, {}).keys())
                print("Updating third menu with:", third_options)
                return third_options

            def update_textbox(selected_category, selected_option, selected_task):
                task_data = merged_dict.get(selected_category, {}).get(selected_option, {}).get(selected_task, {})
                prompt = task_data.get("prompt", "")
                preview_image = task_data.get("initial_state", "")
                task_hint = "Task Hint: " + task_data.get("hint", "")
                print("Updating textbox with prompt:", prompt, "image:", preview_image, "hint:", task_hint)
                return prompt, preview_image, task_hint

            # 绑定下拉菜单变化的回调
            first_menu.change(fn=update_second_menu, inputs=first_menu, outputs=second_menu)
            second_menu.change(fn=update_third_menu, inputs=[first_menu, second_menu], outputs=third_menu)
            third_menu.change(fn=update_textbox, inputs=[first_menu, second_menu, third_menu], outputs=[image_preview, hintbox])

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Computer Use OOTB...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")

    chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580)

    # 当用户提交消息或点击发送按钮时调用
    def handle_message(user_input: str, chat_history: List[Tuple[str, str]], settings: dict) -> List[Tuple[str, str]]:
        if not user_input.strip():
            print("Empty input received, ignoring.")
            return chat_history
        print("Handling message from user:", user_input)
        updated_history = process_input(user_input, chat_history, settings)
        print("Updated chat history:", updated_history)
        return updated_history

    # 绑定按钮点击和输入提交事件
    submit_button.click(handle_message, inputs=[chat_input, chat_history_state, settings_state], outputs=chat_history_state)
    chat_input.submit(handle_message, inputs=[chat_input, chat_history_state, settings_state], outputs=chat_history_state)

    demo.launch(share=True, allowed_paths=["./"])  # TODO: allowed_paths
