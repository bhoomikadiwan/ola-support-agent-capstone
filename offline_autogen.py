
"""Offline AutoGen review using two deterministic agents."""

import asyncio
import json
from typing import Sequence

from pydantic import BaseModel
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    ChatMessage,
    TextMessage,
    StructuredMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat


class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class PolicyReviewer(BaseChatAgent):
    def __init__(self):
        super().__init__(
            "Policy_Reviewer",
            "Checks a draft against its evidence"
        )

    @property
    def produced_message_types(self):
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token):
        task = next(
            m.content for m in messages
            if m.source == "user"
        )

        payload = json.loads(task)

        approved = (
            payload["draft"].strip()
            == payload["evidence"].strip()
        )

        decision = "APPROVE" if approved else "REVISE"

        return Response(
            chat_message=TextMessage(
                content=decision,
                source=self.name
            )
        )

    async def on_reset(self, cancellation_token):
        pass


class FinalEditor(BaseChatAgent):
    def __init__(self):
        super().__init__(
            "Final_Editor",
            "Produces a structured final verdict"
        )

    @property
    def produced_message_types(self):
        return [StructuredMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token):
        task = next(
            m.content for m in messages
            if m.source == "user"
        )

        payload = json.loads(task)

        review_messages = [
            m.content
            for m in messages
            if m.source == "Policy_Reviewer"
        ]

        approved = (
            len(review_messages) > 0
            and review_messages[-1] == "APPROVE"
        )

        verdict = Verdict(
            approved=approved,
            final_answer=(
                payload["draft"]
                if approved
                else payload["evidence"]
            ),
            reason=(
                "Draft matches evidence"
                if approved
                else "Replaced unsupported draft with grounded evidence"
            )
        )

        return Response(
            chat_message=StructuredMessage(
                content=verdict,
                source=self.name
            )
        )

    async def on_reset(self, cancellation_token):
        pass


async def review(draft: str, evidence: str) -> Verdict:
    team = RoundRobinGroupChat(
        [PolicyReviewer(), FinalEditor()],
        max_turns=2,
        custom_message_types=[StructuredMessage[Verdict]]
    )

    result = await team.run(
        task=json.dumps({
            "draft": draft,
            "evidence": evidence
        })
    )

    final = result.messages[-1]
    return Verdict.model_validate(final.content)


if __name__ == "__main__":
    examples = [
        ("Refunds require review.", "Refunds require review."),
        ("Refunds are always instant.", "Refunds require review.")
    ]

    for draft, evidence in examples:
        verdict = asyncio.run(review(draft, evidence))
        print(verdict.model_dump_json(indent=2))
