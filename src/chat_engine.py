import base64
from typing import Literal
import requests
import os
import json

STANDARD_TOOLS: list[
    dict[
        str,
        str | dict[str, str | dict[str, str | dict[str, dict[str, str]] | list[str]]],
    ]
] = [
    {
        "type": "function",
        "function": {
            "name": "respond_to_user",
            "description": "Use this tool for conversations, questions or tasks that are NOT translations. Respond to the user's voice message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transcription": {
                        "type": "string",
                        "description": "Exact transcription of what the user said in the audio.",
                    },
                    "response": {
                        "type": "string",
                        "description": "Your conversational response to the user. Keep it to 1-4 short sentences.",
                    },
                },
                "required": ["transcription", "response"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_audio",
            "description": "Use this tool only for when the user explicitely wants to have their audio translated. Translate the user's voice message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transcription": {
                        "type": "string",
                        "description": "Exact transcription of what the user said in the audio. Use the original language.",
                    },
                    "target_language": {
                        "type": "string",
                        "description": "The target language of the translation.",
                    },
                    "translation": {
                        "type": "string",
                        "description": "The raw exact translation of what the user said without any other messages.",
                    },
                },
                "required": ["transcription", "target_language", "translation"],
            },
        },
    },
]


class LlamaChatEngine:
    def __init__(
        self,
        server_url=os.environ.get("LLAMA_SERVER_URL", "127.0.0.1:8080"),
        system_prompt="You are a helpful AI assistant.",
        model_name=os.environ.get(
            "MODEL_NAME",
            "ggml-org/gemma-4-E2B-it-GGUF",  # Name usually does not matter for llama.cpp server
        ),
        tools=[],
        save_messages: bool = True,  # TODO: Make to options object to specify more options of the engine
        choose_tool: Literal["respond_to_user", "translate_audio"] = "respond_to_user",
    ):
        """
        Initializes the chat engine and sets the system prompt.
        """
        self.server_url = server_url
        self.endpoint = f"{self.server_url}/v1/chat/completions"
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.tools: list[dict] = STANDARD_TOOLS + tools
        self.messages: list[dict[str, str | list[dict]]] = []
        self.save_messages = save_messages
        self.choose_tool = choose_tool
        self.reset_chat()  # Initialize chat history (including system prompt)

    def reset_chat(self):
        """Clears previous history and restarts with the system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        print("--- Chat history has been reset ---")

    def _encode_audio(self, audio_path):
        """Helper function: reads audio and converts it to Base64."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"The file '{audio_path}' was not found.")

        with open(audio_path, "rb") as audio_file:
            base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        audio_format = audio_path.split(".")[-1].lower()
        if audio_format not in ["wav", "mp3"]:
            audio_format = "wav"

        return base64_audio, audio_format

    def send_message(self, text=None, audio_path=None, temperature=0.4):
        """
        Adds the new message (text and/or audio) to history,
        sends full history to the API, and stores the response.
        """
        if not text and not audio_path:
            return "Error: Either text or audio must be provided."

        # 1. Build content for the new user message
        message_content: list[dict[str, str | dict]] = []

        # If text is provided, append it
        if text:
            message_content.append({"type": "text", "text": text})

        # If audio is provided, process and append it
        if audio_path:
            try:
                b64_audio, a_format = self._encode_audio(audio_path)
                message_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64_audio, "format": a_format},
                    }
                )

                message_content.append(
                    {
                        "type": "text",
                        "text": "The user just spoke to you (audio). Respond to what they said.",
                    }
                )
            except Exception as e:
                return f"Audio error: {e}"

        # 2. Append message to local history
        self.messages.append({"role": "user", "content": message_content})

        # 3. Build payload for the server (with full history)
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": self.messages,
                "temperature": str(temperature),
                "stream": False,
                "return_progress": True,
                "reasoning_format": "auto",
                "backend_sampling": False,
                "timings_per_token": True,
                "tools": self.tools,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": self.choose_tool},
                },
            }
        )
        headers = {"Content-Type": "application/json"}

        # 4. Send API request
        try:
            response = requests.request(
                "POST", url=self.endpoint, headers=headers, data=payload
            )
            response.raise_for_status()

            result_json = response.json()

            # Extract assistant's exact response
            assistant_message = result_json["choices"][0]["message"]

            # 5. Append server response to history!
            # This lets the model "remember" its own response in the next turn.
            self.messages.append(assistant_message)

            # Check whether the model called a tool
            if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                tool_call = assistant_message["tool_calls"][0]
                tool_name = tool_call["function"]["name"]
                arguments_string = tool_call["function"]["arguments"]

                try:
                    parsed_args = json.loads(arguments_string)

                    # Logic based on called tool:
                    if tool_name == "translate_audio":
                        print(
                            f"\n[Tool chosen: translation to {parsed_args.get('target_language')}]"
                        )
                        # Convert to unified format for speech output
                        return {
                            "transcription": parsed_args.get("transcription"),
                            "response": f"The translation is: {parsed_args.get('translation')}",
                        }

                    elif tool_name == "respond_to_user":
                        return (
                            parsed_args  # Return transcription & response directly
                        )

                except json.JSONDecodeError:
                    return {
                        "transcription": "Error",
                        "response": "Could not parse the AI JSON.",
                    }

            if (
                not self.save_messages
            ):  # Reset chat if message history should not be saved
                self.reset_chat()

            # Fallback if the model sends regular text for any reason
            return assistant_message.get("content", "No response and no tool call.")

        except requests.exceptions.RequestException as e:
            # On connection error, remove last message
            # so we can retry without polluting chat history.
            self.messages.pop()
            return f"API communication error: {e}\nDetails: {getattr(e.args[0], 'text', 'No details')}"
