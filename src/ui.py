import logging

import gradio as gr
from groq import APIConnectionError, APIStatusError, RateLimitError

from src.interviewer import ChatMessage, Interviewer
from src.resume import ResumeError, extract_resume_text

LOGGER = logging.getLogger(__name__)


def create_app() -> gr.ChatInterface:
    try:
        interviewer = Interviewer()
        configuration_error: str | None = None
    except ValueError as error:
        interviewer = None
        configuration_error = str(error)

    def chat(
        message: str,
        history: list[ChatMessage],
        role: str,
        level: str,
        interview_type: str,
        feedback_mode: str,
        cv_file: str | None,
    ) -> str:
        if interviewer is None:
            return configuration_error or "The interview coach is not configured."

        try:
            resume_text = extract_resume_text(cv_file)
            return interviewer.respond(
                message,
                history,
                role,
                level,
                interview_type,
                feedback_mode,
                resume_text,
            )
        except ResumeError as error:
            return str(error)
        except RateLimitError:
            return "Groq is temporarily rate-limited. Please wait a moment and retry."
        except APIConnectionError:
            return "I could not reach Groq. Check your internet connection and retry."
        except APIStatusError as error:
            LOGGER.exception("Groq returned status %s", error.status_code)
            return "Groq could not process that request. Please retry."
        except Exception:
            LOGGER.exception("Unexpected interview coach error")
            return "Something unexpected went wrong. Please retry."

    return gr.ChatInterface(
        fn=chat,
        title="Suhair — AI Interview Coach",
        description=(
            "Choose Real interview for an authentic simulation or Guided "
            "practice for feedback after every answer. Upload a CV for "
            "personalized questions. Type \"End interview\" at any time to "
            "receive a full evaluation. Uploaded CV text is sent to Groq."
        ),
        additional_inputs=[
            gr.Textbox(
                value="Software Engineer",
                label="Target role",
                placeholder="e.g. Product Manager",
            ),
            gr.Dropdown(
                choices=["Entry level", "Mid level", "Senior", "Lead / Manager"],
                value="Mid level",
                label="Experience level",
            ),
            gr.Dropdown(
                choices=["Mixed", "Behavioral", "Technical", "Case study"],
                value="Mixed",
                label="Interview type",
            ),
            gr.Dropdown(
                choices=["Real interview", "Guided practice"],
                value="Guided practice",
                label="Feedback mode",
                info="Choose when you want to receive coaching.",
            ),
            gr.File(
                label="CV (optional)",
                file_types=[".pdf", ".docx", ".txt"],
                file_count="single",
                type="filepath",
            ),
        ],
        additional_inputs_accordion="Interview profile",
        examples=[
            [
                "Start the interview.",
                "Software Engineer",
                "Mid level",
                "Mixed",
                "Guided practice",
                None,
            ],
            [
                "I want to practice leadership questions.",
                "Engineering Manager",
                "Lead / Manager",
                "Behavioral",
                "Guided practice",
                None,
            ],
            [
                "End interview and give me my final evaluation.",
                "Software Engineer",
                "Mid level",
                "Mixed",
                "Real interview",
                None,
            ],
        ],
        flagging_mode="never",
        save_history=True,
    )
