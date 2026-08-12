"""Abstract base class for SDFT rollout environments."""

from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """One training example's rollout lifecycle.

    Subclasses must implement run() which populates:
        - completion_text: model-generated completion
        - privileged_information_prompt: teacher prompt with privileged info
    """

    prompt_text: str
    vllm_base_url: str
    completion_text: str | None
    privileged_information_prompt: str | None

    @abstractmethod
    def run(self) -> None:
        """Run full episode. Must populate completion_text and privileged_information_prompt."""
        ...
