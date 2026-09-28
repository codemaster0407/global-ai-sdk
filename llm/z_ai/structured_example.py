from pydantic import BaseModel, Field
from llm.z_ai.inference import llm_structured_response


class Task(BaseModel):
    title: str = Field(description='Short title of the task')
    priority: str = Field(description='One of: low, medium, high')
    estimated_hours: float


class MeetingSummary(BaseModel):
    summary: str = Field(description='Two sentence summary of the meeting')
    participants: list[str]
    tasks: list[Task]


if __name__ == '__main__':
    transcript = (
        'John: We need the login page done by Friday, that is top priority. '
        'Priya: I can take it, maybe 6 hours. Also someone should update the docs. '
        'John: Sam, can you do docs? Low priority, 2 hours tops.'
    )

    result = llm_structured_response(
        output_model=MeetingSummary,
        system_prompt='You extract structured meeting notes from transcripts.',
        user_prompt=transcript,
    )

    print(type(result).__name__)
    print(result.summary)
    for task in result.tasks:
        print(f'- {task.title} [{task.priority}] {task.estimated_hours}h')
