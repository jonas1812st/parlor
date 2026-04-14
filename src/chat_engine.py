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
        system_prompt="Du bist ein hilfreicher KI-Assistent.",
        model_name=os.environ.get(
            "MODEL_NAME",
            "ggml-org/gemma-4-E2B-it-GGUF",  # Name ist meist egal beim llama.cpp server
        ),
        tools=[],
        save_messages: bool = True,  # TODO: Make to options object to specify more options of the engine
        choose_tool: Literal["respond_to_user", "translate_audio"] = "respond_to_user",
    ):
        """
        Initialisiert die Chat-Engine und legt den System-Prompt fest.
        """
        self.server_url = server_url
        self.endpoint = f"{self.server_url}/v1/chat/completions"
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.tools: list[dict] = STANDARD_TOOLS + tools
        self.messages: list[dict[str, str | list[dict]]] = []
        self.save_messages = save_messages
        self.choose_tool = choose_tool
        self.reset_chat()  # Setzt den Chatverlauf initial auf (inkl. System-Prompt)

    def reset_chat(self):
        """Löscht den bisherigen Verlauf und startet neu mit dem System-Prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        print("--- Chatverlauf wurde zurückgesetzt ---")

    def _encode_audio(self, audio_path):
        """Hilfsfunktion: Liest Audio ein und wandelt es in Base64 um."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Die Datei '{audio_path}' wurde nicht gefunden.")

        with open(audio_path, "rb") as audio_file:
            base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        audio_format = audio_path.split(".")[-1].lower()
        if audio_format not in ["wav", "mp3"]:
            audio_format = "wav"

        return base64_audio, audio_format

    def send_message(self, text=None, audio_path=None, temperature=0.4):
        """
        Fügt die neue Nachricht (Text und/oder Audio) dem Verlauf hinzu,
        sendet den gesamten Verlauf an die API und speichert die Antwort.
        """
        if not text and not audio_path:
            return "Fehler: Es muss entweder Text oder Audio übergeben werden."

        # 1. Inhalt der neuen Benutzernachricht zusammenbauen
        message_content: list[dict[str, str | dict]] = []

        # Falls Text vorhanden ist, hinzufügen
        if text:
            message_content.append({"type": "text", "text": text})

        # Falls Audio vorhanden ist, verarbeiten und hinzufügen
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
                        "text": "Der Benutzer hat gerade mit dir gesprochen (Audio). Antworte darauf, was er sagt.",
                    }
                )
            except Exception as e:
                return f"Audio-Fehler: {e}"

        # 2. Nachricht an unseren lokalen Verlauf anhängen
        self.messages.append({"role": "user", "content": message_content})

        # 3. Payload für den Server erstellen (mit gesamtem Verlauf)
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

        # 4. API Request senden
        try:
            response = requests.request(
                "POST", url=self.endpoint, headers=headers, data=payload
            )
            response.raise_for_status()

            result_json = response.json()

            # Die exakte Antwort des Assistenten extrahieren
            assistant_message = result_json["choices"][0]["message"]

            # 5. Die Antwort des Servers an unseren Verlauf anhängen!
            # Dadurch "erinnert" sich das Modell bei der nächsten Frage an seine eigene Antwort.
            self.messages.append(assistant_message)

            # Prüfen, ob das Modell das Tool aufgerufen hat
            if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                tool_call = assistant_message["tool_calls"][0]
                tool_name = tool_call["function"]["name"]
                arguments_string = tool_call["function"]["arguments"]

                try:
                    parsed_args = json.loads(arguments_string)

                    # Logik je nach aufgerufenem Tool:
                    if tool_name == "translate_audio":
                        print(
                            f"\n[Tool gewählt: Übersetzung ins {parsed_args.get('target_language')}]"
                        )
                        # Wir wandeln es für die Sprachausgabe so um, dass es einheitlich bleibt
                        return {
                            "transcription": parsed_args.get("transcription"),
                            "response": f"Die Übersetzung lautet: {parsed_args.get('translation')}",
                        }

                    elif tool_name == "respond_to_user":
                        return (
                            parsed_args  # Gibt einfach transcription & response zurück
                        )

                except json.JSONDecodeError:
                    return {
                        "transcription": "Fehler",
                        "response": "Konnte das JSON der KI nicht parsen.",
                    }

            if (
                not self.save_messages
            ):  # Reset chat if message history should not be saved
                self.reset_chat()

            # Fallback, falls das Modell (warum auch immer) normalen Text schickt
            return assistant_message.get("content", "Keine Antwort und kein Tool-Call.")

        except requests.exceptions.RequestException as e:
            # Bei einem Verbindungsfehler entfernen wir unsere letzte Nachricht wieder,
            # damit wir es noch einmal versuchen können, ohne den Chatverlauf zu vergiften.
            self.messages.pop()
            return f"API-Kommunikationsfehler: {e}\nDetails: {getattr(e.args[0], 'text', 'Keine Details')}"
