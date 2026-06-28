const DEFAULT_TITLE_TOKEN_LIMIT = 9;

export function titleFromQuestion(
  question: string,
  tokenLimit: number = DEFAULT_TITLE_TOKEN_LIMIT
): string {
  const tokens = question.trim().split(/\s+/).filter((token) => token.length > 0);
  if (tokens.length <= tokenLimit) {
    return tokens.join(" ");
  }
  return `${tokens.slice(0, tokenLimit).join(" ")}…`;
}
