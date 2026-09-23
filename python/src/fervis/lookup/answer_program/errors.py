class AnswerProgramContractError(ValueError):
    """A stable fail-closed answer-program contract outcome."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class UnsupportedAnswerProgramSchema(AnswerProgramContractError):
    """Readable historical evidence does not imply an executable old contract."""

    def __init__(self) -> None:
        super().__init__("incompatible_program_schema", "unsupported answer-program schema revision")
